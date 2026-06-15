//  Root component. Fetches all data from FastAPI on mount

import { useState, useEffect } from "react";
import { api } from "./apiClient.js";
import SalesCastDashboard from "./SalesCastDashboard.jsx";

const POLL_INTERVAL_MS = 5 * 60 * 1000; // refresh every 5 minutes

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  async function fetchAll() {
    try {
      const [kpis, forecast, actuals, segments, insights, models, health] =
        await Promise.all([
          api.kpis(),
          api.forecast("ensemble", 90),
          api.actuals("daily"),
          api.segments(),
          api.insights(),
          api.models(),
          api.health(),
        ]);
      setData({ kpis, forecast, actuals, segments, insights, models, health });
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          flexDirection: "column",
          gap: 16,
        }}
      >
        <div
          style={{
            width: 32,
            height: 32,
            border: "2px solid #21262D",
            borderTop: "2px solid #58A6FF",
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
          }}
        />
        <span style={{ color: "#8B949E", fontSize: 13 }}>
          Loading SalesCast…
        </span>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          flexDirection: "column",
          gap: 12,
        }}
      >
        <span style={{ fontSize: 24 }}>⚠</span>
        <span style={{ color: "#F85149", fontSize: 14 }}>
          Could not connect to the API
        </span>
        <code
          style={{
            color: "#8B949E",
            fontSize: 12,
            background: "#161B22",
            padding: "4px 10px",
            borderRadius: 5,
          }}
        >
          {error}
        </code>
        <span style={{ color: "#8B949E", fontSize: 12 }}>
          Make sure the FastAPI backend is running on port 8000
        </span>
        <button
          onClick={fetchAll}
          style={{
            marginTop: 8,
            padding: "6px 16px",
            borderRadius: 6,
            border: "1px solid #30363D",
            background: "transparent",
            color: "#E6EDF3",
            cursor: "pointer",
            fontSize: 13,
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  return <SalesCastDashboard data={data} onRefresh={fetchAll} />;
}
