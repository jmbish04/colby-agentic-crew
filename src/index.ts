import { Hono } from 'hono';
import { Ai } from '@cloudflare/ai';

// Define our environment bindings
type Bindings = {
	AI: Ai;
	DB: D1Database;
	VECTOR_INDEX: VectorizeIndex;
	// SECRET_KEY: string; // For auth
};

const app = new Hono<{ Bindings: Bindings }>();

// --- 1. INGESTION ENDPOINT ---
// Called by your local Python script for each file
app.post('/api/v1/process', async (c) => {
	const body = await c.req.json<{
		repo_full_name: string;
		file_path: string;
		content: string;
	}>();

	// Simple auth check
	// const auth = c.req.header('Authorization');
	// if (auth !== `Bearer ${c.env.SECRET_KEY}`) {
	// 	return c.json({ error: 'Unauthorized' }, 401);
	// }

	// Create hash to check for duplicates
	const content_hash = await sha256(body.content);
	const repo_url = `https://github.com/${body.repo_full_name}/blob/main/${body.file_path}`; // Assumes main branch

	// Check if this exact file version is already processed
	const { count } = await c.env.DB.prepare(
		'SELECT count(*) as count FROM code_artifacts WHERE repo_full_name = ? AND file_path = ? AND content_hash = ?'
	)
		.bind(body.repo_full_name, body.file_path, content_hash)
		.first<{ count: number }>();

	if (count > 0) {
		return c.json({ message: 'Already processed' }, 200);
	}

	const id = crypto.randomUUID();
	const content_snippet = body.content.substring(0, 200) + '...';

	// 1. Get AI analysis (Summary, Tags, Use Case)
	const ai = new Ai(c.env.AI);
	const prompt = `
		You are a Cloudflare expert. Analyze the following code file and respond *only* with a valid JSON object.
		File: ${body.repo_full_name}/${body.file_path}
		Content:
		\`\`\`
		${body.content.substring(0, 4000)}
		\`\`\`

		Provide:
		1. "summary": A brief summary of what this code does.
		2. "tags": An array of strings. Tags can include "best-practice", "deprecation", "d1", "kv", "durable-objects", "queues", "ai", "hono", "react", "example", "template", etc.
		3. "use_case": A short description of the problem this code solves or the pattern it demonstrates.
	`;

	const ai_result_text = await ai.run('@cf/mistral/mistral-7b-instruct-v0.1', { prompt });
	let ai_data = { summary: 'AI summary failed', tags: ['parse-error'], use_case: 'AI summary failed' };
	try {
		ai_data = JSON.parse(ai_result_text as string);
	} catch (e) {
		console.error('Failed to parse AI JSON response:', ai_result_text);
	}

	// 2. Get AI Embedding
	const { data: embeddings } = await ai.run('@cf/baai/bge-base-en-v1.5', { text: [body.content] });

	// 3. Save to D1 and Vectorize in a transaction
	const { success } = await c.env.DB.batch([
		c.env.DB.prepare(
			'INSERT INTO code_artifacts (id, repo_full_name, file_path, repo_url, content_hash, content_snippet, ai_summary, ai_tags, ai_use_case) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'
		).bind(
			id,
			body.repo_full_name,
			body.file_path,
			repo_url,
			content_hash,
			content_snippet,
			ai_data.summary,
			JSON.stringify(ai_data.tags),
			ai_data.use_case
		),
	]);

    if (success) {
        await c.env.VECTOR_INDEX.insert([{ id: id, values: embeddings[0] }]);
    } else {
        return c.json({ error: 'Failed to write to D1' }, 500);
    }

	return c.json({ id: id, message: 'Processed and embedded' }, 201);
});

// --- 2. REPO CHECK ENDPOINT ---
// Called by import_cloudflare.py to avoid re-downloading
app.get('/api/v1/check', async (c) => {
	const repo_full_name = c.req.query('repo_full_name');
	if (!repo_full_name) {
		return c.json({ error: 'repo_full_name query param required' }, 400);
	}

	const { count } = await c.env.DB.prepare(
		'SELECT count(*) as count FROM code_artifacts WHERE repo_full_name = ?'
	)
		.bind(repo_full_name)
		.first<{ count: number }>();

	if (count > 0) {
		return c.json({ status: 'exists' }, 200); // 200 OK means "we have it"
	}
	return c.json({ status: 'not_found' }, 404); // 404 Not Found means "we don't have it, go get it"
});

// --- 3. RAG QUERY ENDPOINT ---
// Your "expert consultant" agent
app.post('/api/v1/rag_query', async (c) => {
	const { query } = await c.req.json<{ query: string }>();

	const ai = new Ai(c.env.AI);

	// 1. Get embedding for the query
	const { data: embeddings } = await ai.run('@cf/baai/bge-base-en-v1.5', { text: [query] });
	
	// 2. Query Vectorize
	const vector_matches = await c.env.VECTOR_INDEX.query(embeddings[0], { topK: 5 });
	const match_ids = vector_matches.map((v) => v.id);

	if (match_ids.length === 0) {
		return c.json({ answer: "I couldn't find any relevant code snippets in my knowledge base to answer that." });
	}

	// 3. Get metadata from D1 for the vector matches
	const { results } = await c.env.DB.prepare(
		`SELECT repo_full_name, file_path, content_snippet, ai_summary, ai_use_case FROM code_artifacts WHERE id IN (${match_ids.map(() => '?').join(',')})`
	).bind(...match_ids).all();

	// 4. Build context for the LLM
	const context = (results as any[])
		.map(
			(r, i) =>
				`Context ${i + 1}:
Source: ${r.repo_full_name}/${r.file_path}
Use Case: ${r.ai_use_case}
Summary: ${r.ai_summary}
Snippet: ${r.content_snippet}\n`
		)
		.join('\n---\n');

	// 5. Ask the LLM to synthesize an answer
	const prompt = `
		You are an expert Cloudflare consultant and part of the 'colby' agentic ecosystem.
		Your task is to answer the user's query based *only* on the provided context from a knowledge base of code artifacts.
		Do not make up information. If the context isn't sufficient, say so.
		
		Context:
		${context}

		User Query:
		${query}

		Answer:
	`;

	const answer = await ai.run('@cf/mistral/mistral-7b-instruct-v0.1', { prompt });
	
	return c.json({ answer: answer, sources: results });
});

// Helper for SHA-256
async function sha256(message: string) {
	const data = new TextEncoder().encode(message);
	const hashBuffer = await crypto.subtle.digest('SHA-256', data);
	const hashArray = Array.from(new Uint8Array(hashBuffer));
	return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

export default app;
