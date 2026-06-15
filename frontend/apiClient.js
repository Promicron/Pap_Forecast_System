// Thin fetch client for the SalesCast FastAPI backend.
const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API POST ${path} → ${res.status}`);
  return res.json();
}

export const api = {
  health: () => get("/health"),
  kpis: () => get("/api/v1/kpis"),
  forecast: (model, horizon) =>
    get(`/api/v1/forecast?model=${model}&horizon=${horizon}`),
  actuals: (granularity) => get(`/api/v1/actuals?granularity=${granularity}`),
  segments: () => get("/api/v1/segments"),
  insights: () => get("/api/v1/insights"),
  models: () => get("/api/v1/models"),
};
