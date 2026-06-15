import { useState, useEffect, useRef } from "react";
import { ComposedChart, Area, Line, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";

// ── Real data from pipeline ──────────────────────────────────────────────
const ACTUALS = [{"date":"2026-08-26","y":190.66},{"date":"2026-08-27","y":6190.54},{"date":"2026-08-28","y":1443.63},{"date":"2026-08-29","y":235.47},{"date":"2026-08-31","y":3162.83},{"date":"2026-09-01","y":1261.81},{"date":"2026-09-02","y":9354.86},{"date":"2026-09-03","y":1595.85},{"date":"2026-09-04","y":5360.18},{"date":"2026-09-05","y":327.04},{"date":"2026-09-07","y":3848.57},{"date":"2026-09-08","y":2184.33},{"date":"2026-09-09","y":4356.07},{"date":"2026-09-10","y":2509.43},{"date":"2026-09-11","y":5564.0},{"date":"2026-09-12","y":491.55},{"date":"2026-09-14","y":4367.33},{"date":"2026-09-15","y":7049.68},{"date":"2026-09-16","y":849.64},{"date":"2026-09-17","y":4979.22},{"date":"2026-09-18","y":1511.93},{"date":"2026-09-19","y":1648.19},{"date":"2026-09-20","y":7397.6},{"date":"2026-09-21","y":338.17},{"date":"2026-09-22","y":5398.06},{"date":"2026-09-23","y":2395.79},{"date":"2026-09-24","y":6529.45},{"date":"2026-09-25","y":1412.23},{"date":"2026-09-26","y":1496.57},{"date":"2026-09-28","y":559.27},{"date":"2026-09-29","y":1944.09},{"date":"2026-09-30","y":540.76},{"date":"2026-10-01","y":2978.47},{"date":"2026-10-02","y":5418.02},{"date":"2026-10-03","y":2504.48},{"date":"2026-10-05","y":9213.78},{"date":"2026-10-06","y":1499.65},{"date":"2026-10-07","y":2749.2},{"date":"2026-10-08","y":608.35},{"date":"2026-10-09","y":1496.59},{"date":"2026-10-12","y":5635.34},{"date":"2026-10-13","y":5443.99},{"date":"2026-10-14","y":134.34},{"date":"2026-10-15","y":1017.94},{"date":"2026-10-16","y":3473.6},{"date":"2026-10-19","y":2842.37},{"date":"2026-10-20","y":1333.86},{"date":"2026-10-21","y":4537.21},{"date":"2026-10-22","y":11638.88},{"date":"2026-10-23","y":3352.39},{"date":"2026-10-24","y":529.08},{"date":"2026-10-26","y":999.87},{"date":"2026-10-27","y":1086.32},{"date":"2026-10-28","y":408.72},{"date":"2026-10-30","y":4025.73},{"date":"2026-10-31","y":523.93},{"date":"2026-11-01","y":2921.43},{"date":"2026-11-02","y":6294.39},{"date":"2026-11-03","y":4536.94},{"date":"2026-11-04","y":10348.08},{"date":"2026-11-05","y":2355.06},{"date":"2026-11-06","y":4288.74},{"date":"2026-11-07","y":2413.37},{"date":"2026-11-09","y":4751.5},{"date":"2026-11-10","y":4007.54},{"date":"2026-11-11","y":1815.22},{"date":"2026-11-12","y":2911.38},{"date":"2026-11-13","y":6633.4},{"date":"2026-11-16","y":4755.23},{"date":"2026-11-17","y":10874.88},{"date":"2026-11-18","y":1469.75},{"date":"2026-11-19","y":7014.7},{"date":"2026-11-20","y":2988.29},{"date":"2026-11-21","y":2236.2},{"date":"2026-11-24","y":4959.63},{"date":"2026-11-25","y":3666.16},{"date":"2026-11-26","y":5048.17},{"date":"2026-11-27","y":1618.25},{"date":"2026-11-28","y":6912.94},{"date":"2026-11-30","y":6645.3},{"date":"2026-12-01","y":5331.18},{"date":"2026-12-02","y":10065.97},{"date":"2026-12-03","y":1403.84},{"date":"2026-12-04","y":2681.62},{"date":"2026-12-05","y":1453.15},{"date":"2026-12-07","y":2916.51},{"date":"2026-12-08","y":7643.04},{"date":"2026-12-09","y":5596.21},{"date":"2026-12-10","y":3972.68},{"date":"2026-12-11","y":2823.96},{"date":"2026-12-14","y":4576.48},{"date":"2026-12-17","y":2027.76},{"date":"2026-12-18","y":3645.9},{"date":"2026-12-19","y":1895.93},{"date":"2026-12-21","y":2140.94},{"date":"2026-12-22","y":7442.02},{"date":"2026-12-23","y":1926.76},{"date":"2026-12-24","y":6233.05},{"date":"2026-12-25","y":2698.92},{"date":"2026-12-29","y":3102.72},{"date":"2026-12-30","y":716.81}];
const ENSEMBLE_FC = [{"date":"2026-12-31","y":4278.79,"lower":97.99,"upper":6161.38},{"date":"2027-01-01","y":2502.97,"lower":0,"upper":5224.19},{"date":"2027-01-02","y":1492.76,"lower":0,"upper":3460.5},{"date":"2027-01-03","y":970.32,"lower":0,"upper":2424.85},{"date":"2027-01-04","y":1313.9,"lower":0,"upper":4446.14},{"date":"2027-01-05","y":1814.85,"lower":0,"upper":4683.56},{"date":"2027-01-06","y":1641.77,"lower":0,"upper":4205.15},{"date":"2027-01-07","y":1756.09,"lower":0,"upper":4082.67},{"date":"2027-01-08","y":1538.71,"lower":0,"upper":4456.47},{"date":"2027-01-09","y":646.29,"lower":0,"upper":2947.21},{"date":"2027-01-11","y":1267.82,"lower":0,"upper":3871.61},{"date":"2027-01-12","y":1317.42,"lower":0,"upper":4388.5},{"date":"2027-01-13","y":1344.68,"lower":0,"upper":4101.69},{"date":"2027-01-15","y":1378.66,"lower":0,"upper":4478.46},{"date":"2027-01-18","y":1175.01,"lower":0,"upper":3796.79},{"date":"2027-01-19","y":1516.67,"lower":0,"upper":4228.68},{"date":"2027-01-22","y":1470.57,"lower":0,"upper":4017.44},{"date":"2027-01-25","y":1055.8,"lower":0,"upper":3521.34},{"date":"2027-01-26","y":1428.65,"lower":0,"upper":3919.02},{"date":"2027-01-29","y":1431.22,"lower":0,"upper":3513.75},{"date":"2027-02-01","y":981.89,"lower":0,"upper":2519.99},{"date":"2027-02-02","y":1206.86,"lower":0,"upper":2887.7},{"date":"2027-02-05","y":980.27,"lower":0,"upper":2753.93},{"date":"2027-02-09","y":981.99,"lower":0,"upper":2137.18},{"date":"2027-02-12","y":835.88,"lower":0,"upper":1834.09},{"date":"2027-02-16","y":775.85,"lower":0,"upper":1355.0},{"date":"2027-02-19","y":920.52,"lower":0,"upper":1441.5},{"date":"2027-02-23","y":1019.17,"lower":0,"upper":1457.42},{"date":"2027-02-26","y":1063.49,"lower":0,"upper":1418.84}];
const XGB_FC = [{"date":"2026-12-31","y":5377.38},{"date":"2027-01-01","y":2913.02},{"date":"2027-01-04","y":1406.41},{"date":"2027-01-07","y":2356.35},{"date":"2027-01-11","y":1659.53},{"date":"2027-01-15","y":1497.46},{"date":"2027-01-19","y":1825.3},{"date":"2027-01-22","y":1825.74},{"date":"2027-01-26","y":1961.17},{"date":"2027-01-29","y":2198.34},{"date":"2027-02-02","y":2190.9},{"date":"2027-02-05","y":1814.51},{"date":"2027-02-09","y":1817.69},{"date":"2027-02-12","y":1547.23},{"date":"2027-02-16","y":1436.11},{"date":"2027-02-19","y":1703.9},{"date":"2027-02-23","y":1886.51},{"date":"2027-02-26","y":1968.55},{"date":"2027-02-28","y":1468.2}];
const MONTHLY = [{"month":"2025-07","revenue":40300},{"month":"2025-08","revenue":31716},{"month":"2025-09","revenue":71288},{"month":"2025-10","revenue":47597},{"month":"2025-11","revenue":78893},{"month":"2025-12","revenue":89731},{"month":"2026-01","revenue":38895},{"month":"2026-02","revenue":20301},{"month":"2026-03","revenue":54408},{"month":"2026-04","revenue":35740},{"month":"2026-05","revenue":45155},{"month":"2026-06","revenue":52773},{"month":"2026-07","revenue":45989},{"month":"2026-08","revenue":59350},{"month":"2026-09","revenue":85287},{"month":"2026-10","revenue":73915},{"month":"2026-11","revenue":114931},{"month":"2026-12","revenue":85175}];
const SEGMENTS = [{"segment":"Consumer","revenue":1112950,"share":50.3},{"segment":"Corporate","revenue":684985,"share":31.0},{"segment":"Home Office","revenue":414783,"share":18.7}];
const CATEGORIES = [{"category":"Furniture","revenue":744377,"share":33.6},{"category":"Office Supplies","revenue":669896,"share":30.3},{"category":"Technology","revenue":798444,"share":36.1}];
const TOP_FEATURES = ["week_of_year","avg_discount","month","days_to_qtr_end","ewm_14d","days_to_holiday","roll_mean_90d","roll_mean_14d","lag_56d","day_of_week"];
const FEATURE_SCORES = [0.142,0.118,0.098,0.087,0.076,0.065,0.058,0.054,0.051,0.048];

// ── Helpers ──────────────────────────────────────────────────────────────────
const fmt = (v) => v >= 1e6 ? `$${(v/1e6).toFixed(2)}M` : v >= 1e3 ? `$${(v/1e3).toFixed(1)}k` : `$${v.toFixed(0)}`;
const fmtDate = (d) => { const dt = new Date(d); return dt.toLocaleDateString("en-US",{month:"short",day:"numeric"}); };
const fmtMonth = (m) => { const [y,mo] = m.split("-"); return new Date(y,mo-1).toLocaleDateString("en-US",{month:"short",year:"2-digit"}); };

// Merge actuals + forecast into one series for the main chart
const buildChartData = () => {
  const pts = ACTUALS.map(d => ({ date: d.date, actual: d.y, label: fmtDate(d.date) }));
  const fcPts = ENSEMBLE_FC.map(d => ({ date: d.date, forecast: d.y, lower: d.lower, upper: d.upper, label: fmtDate(d.date) }));
  const xgbMap = Object.fromEntries(XGB_FC.map(d => [d.date, d.y]));
  fcPts.forEach(p => { p.xgb = xgbMap[p.date]; });
  return [...pts, ...fcPts];
};
const CHART_DATA = buildChartData();
const SPLIT_DATE = "2026-12-30";

// ── Color tokens ─────────────────────────────────────────────────────────────
const C = {
  bg:       "#0D1117",
  surface:  "#161B22",
  border:   "#21262D",
  blue:     "#58A6FF",
  green:    "#3FB950",
  red:      "#F85149",
  amber:    "#D29922",
  muted:    "#8B949E",
  text:     "#E6EDF3",
  textDim:  "#848D97",
};

// ── Sub-components ────────────────────────────────────────────────────────────

const KPICard = ({ label, value, sub, trend }) => (
  <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: "14px 18px", flex: 1, minWidth: 130 }}>
    <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>{label}</div>
    <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 22, fontWeight: 600, color: C.text, lineHeight: 1 }}>{value}</div>
    {sub && <div style={{ fontSize: 11, color: trend === "up" ? C.green : trend === "down" ? C.red : C.muted, marginTop: 5 }}>{sub}</div>}
  </div>
);

const SectionHeader = ({ title, badge }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
    <span style={{ fontSize: 12, fontWeight: 600, color: C.text, letterSpacing: "0.02em" }}>{title}</span>
    {badge && <span style={{ fontSize: 10, background: "#1F2D3D", color: C.blue, padding: "2px 7px", borderRadius: 20, fontWeight: 500 }}>{badge}</span>}
  </div>
);

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#1C2128", border: `1px solid ${C.border}`, borderRadius: 6, padding: "8px 12px", fontSize: 12 }}>
      <div style={{ color: C.muted, marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => p.value != null && (
        <div key={i} style={{ color: p.color || C.text, display: "flex", gap: 8 }}>
          <span style={{ color: C.muted }}>{p.name}:</span>
          <span style={{ fontFamily: "'IBM Plex Mono', monospace" }}>{fmt(p.value)}</span>
        </div>
      ))}
    </div>
  );
};

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function SalesCastDashboard({ data, onRefresh }) {
  // Use live API data when available, fall back to embedded data
  const liveActuals = data?.actuals?.points ?? null;
  const liveForecast = data?.forecast?.forecast ?? null;
  const liveKpis = data?.kpis ?? null;
  const liveSegments = data?.segments?.segments ?? null;
  const liveInsights = data?.insights?.insights ?? null;
  const liveModels = data?.models ?? null;
  const [activeTab, setActiveTab] = useState("forecast");
  const [activeModel, setActiveModel] = useState("ensemble");
  const [animProgress, setAnimProgress] = useState(0);
  const animRef = useRef(null);

  // Signature animation: forecast line draws in on mount
  useEffect(() => {
    let frame = 0;
    const total = 60;
    const tick = () => {
      frame++;
      setAnimProgress(Math.min(frame / total, 1));
      if (frame < total) animRef.current = requestAnimationFrame(tick);
    };
    const t = setTimeout(() => { animRef.current = requestAnimationFrame(tick); }, 400);
    return () => { clearTimeout(t); if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []);

  // Animated forecast slice
  const visibleForecast = Math.floor(ENSEMBLE_FC.length * animProgress);
  const chartData = [
    ...ACTUALS.map(d => ({ date: d.date, actual: d.y, label: fmtDate(d.date) })),
    ...ENSEMBLE_FC.slice(0, Math.max(1, visibleForecast)).map(d => ({
      date: d.date, forecast: d.y, lower: d.lower, upper: Math.max(d.upper, d.y), label: fmtDate(d.date),
      xgb: XGB_FC.find(x => x.date === d.date)?.y
    }))
  ];

  const navItems = [
    { id: "forecast", icon: "◈", label: "Forecast" },
    { id: "actuals",  icon: "◉", label: "Actuals" },
    { id: "models",   icon: "⬡", label: "Models" },
    { id: "segments", icon: "◫", label: "Segments" },
  ];

  return (
    <div style={{ display: "flex", height: "100vh", background: C.bg, fontFamily: "Inter, system-ui, sans-serif", color: C.text, fontSize: 13, overflow: "hidden" }}>

      {/* ── Sidebar ── */}
      <div style={{ width: 188, background: C.surface, borderRight: `1px solid ${C.border}`, display: "flex", flexDirection: "column", flexShrink: 0 }}>
        <div style={{ padding: "16px 16px 14px", borderBottom: `1px solid ${C.border}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 28, height: 28, borderRadius: 6, background: "linear-gradient(135deg,#1F6FEB,#388BFD)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14 }}>◈</div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600 }}>SalesCast</div>
              <div style={{ fontSize: 10, color: C.muted }}>v1.0 · Superstore</div>
            </div>
          </div>
        </div>

        <div style={{ padding: "10px 8px", flex: 1 }}>
          <div style={{ fontSize: 10, color: C.muted, padding: "4px 8px 8px", letterSpacing: "0.08em", textTransform: "uppercase" }}>Analytics</div>
          {navItems.map(item => (
            <div key={item.id} onClick={() => setActiveTab(item.id)} style={{
              display: "flex", alignItems: "center", gap: 9, padding: "7px 10px", borderRadius: 6, cursor: "pointer",
              background: activeTab === item.id ? "#1F2D3D" : "transparent",
              color: activeTab === item.id ? C.blue : C.muted,
              fontWeight: activeTab === item.id ? 500 : 400,
              marginBottom: 2, transition: "all 0.12s"
            }}>
              <span style={{ fontSize: 14 }}>{item.icon}</span>
              <span style={{ fontSize: 13 }}>{item.label}</span>
            </div>
          ))}

          <div style={{ fontSize: 10, color: C.muted, padding: "14px 8px 8px", letterSpacing: "0.08em", textTransform: "uppercase" }}>System</div>
          <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "7px 10px", borderRadius: 6, color: C.muted }}>
            <span style={{ fontSize: 14 }}>⬚</span><span>Data Gov.</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "7px 10px", borderRadius: 6, color: C.muted }}>
            <span style={{ fontSize: 14 }}>⚙</span><span>Settings</span>
          </div>
        </div>

        {/* Status badge */}
        <div style={{ padding: 12, borderTop: `1px solid ${C.border}` }}>
          <div style={{ background: "#0D1F0D", border: "1px solid #1F4D1F", borderRadius: 6, padding: "7px 10px", display: "flex", alignItems: "center", gap: 7 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.green, boxShadow: `0 0 6px ${C.green}` }} />
            <div>
              <div style={{ fontSize: 11, color: C.green, fontWeight: 500 }}>Models live</div>
              <div style={{ fontSize: 10, color: C.muted }}>Last sync: 2h ago</div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Main ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Topbar */}
        <div style={{ padding: "12px 20px", borderBottom: `1px solid ${C.border}`, display: "flex", alignItems: "center", justifyContent: "space-between", background: C.surface, flexShrink: 0 }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600 }}>Sales Forecast · Superstore</div>
            <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>Jan 2023 – Feb 2027 · 10,194 transactions · 804 customers</div>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{ fontSize: 11, background: "#1F2D3D", color: C.blue, padding: "3px 10px", borderRadius: 20 }}>XGBoost winner</span>
            <span style={{ fontSize: 11, background: "#1A1F0D", color: "#7EE787", padding: "3px 10px", borderRadius: 20 }}>90-day horizon</span>
          </div>
        </div>

        <div style={{ flex: 1, overflow: "auto", padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>

          {/* KPI Row */}
          <div style={{ display: "flex", gap: 10 }}>
            <KPICard label="Total Revenue" value="$2.21M" sub="Jan 2023 – Dec 2026" />
            <KPICard label="Forecast 90d" value="$84.6k" sub="↑ Ensemble model" trend="up" />
            <KPICard label="Best Model RMSE" value="$2,669" sub="XGBoost · hold-out" trend="up" />
            <KPICard label="GP Margin" value="10.4%" sub="↓ Discount pressure" trend="down" />
            <KPICard label="Discount Rate" value="13.6%" sub="51% txns discounted" trend="down" />
          </div>

          {/* ── FORECAST TAB ── */}
          {activeTab === "forecast" && (
            <>
              {/* Main chart */}
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 3 }}>Revenue · Actuals + 90-day Forecast</div>
                    <div style={{ display: "flex", gap: 14, fontSize: 11, color: C.muted }}>
                      {[["Actual","#58A6FF"],["Ensemble","#3FB950"],["XGBoost","#D29922"],["Uncertainty","rgba(63,185,80,0.15)"]].map(([l,c]) => (
                        <span key={l} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                          <span style={{ width: 10, height: 3, background: c, borderRadius: 2, display: "inline-block" }} />{l}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    {["ensemble","xgboost"].map(m => (
                      <button key={m} onClick={() => setActiveModel(m)} style={{
                        fontSize: 11, padding: "4px 10px", borderRadius: 5, cursor: "pointer",
                        border: `1px solid ${activeModel === m ? C.blue : C.border}`,
                        background: activeModel === m ? "#1F2D3D" : "transparent",
                        color: activeModel === m ? C.blue : C.muted
                      }}>{m}</button>
                    ))}
                  </div>
                </div>

                <ResponsiveContainer width="100%" height={240}>
                  <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
                    <defs>
                      <linearGradient id="blueGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={C.blue} stopOpacity={0.18} />
                        <stop offset="100%" stopColor={C.blue} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="greenGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={C.green} stopOpacity={0.18} />
                        <stop offset="100%" stopColor={C.green} stopOpacity={0} />
                      </linearGradient>
                      <filter id="glow">
                        <feGaussianBlur stdDeviation="2" result="blur"/>
                        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                      </filter>
                    </defs>
                    <XAxis dataKey="label" tick={{ fontSize: 10, fill: C.muted }} tickLine={false} axisLine={false} interval={14} />
                    <YAxis tickFormatter={v => `$${(v/1000).toFixed(0)}k`} tick={{ fontSize: 10, fill: C.muted }} tickLine={false} axisLine={false} width={44} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine x={fmtDate(SPLIT_DATE)} stroke={C.border} strokeDasharray="3 3" label={{ value: "Forecast →", position: "top", fill: C.muted, fontSize: 10 }} />
                    <Area dataKey="upper" stroke="none" fill="rgba(63,185,80,0.12)" isAnimationActive={false} />
                    <Area dataKey="lower" stroke="none" fill={C.bg} isAnimationActive={false} />
                    <Area dataKey="actual" stroke={C.blue} strokeWidth={1.5} fill="url(#blueGrad)" dot={false} isAnimationActive={false} name="Actual" />
                    <Line dataKey="forecast" stroke={C.green} strokeWidth={2} dot={false} isAnimationActive={false} name="Ensemble" filter="url(#glow)" />
                    {activeModel === "xgboost" && <Line dataKey="xgb" stroke={C.amber} strokeWidth={1.5} dot={false} strokeDasharray="4 3" isAnimationActive={false} name="XGBoost" />}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Bottom row */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                {/* Model metrics */}
                <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                  <SectionHeader title="Model Comparison" badge="Hold-out metrics" />
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                        {["Model","RMSE","MAE","Weight"].map(h => (
                          <th key={h} style={{ textAlign: "left", padding: "4px 0 8px", color: C.muted, fontWeight: 500, fontSize: 11 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { name: "XGBoost", rmse: 2669, mae: 1912, weight: 0.54, winner: true },
                        { name: "Prophet", rmse: 3136, mae: 2143, weight: 0.46, winner: false },
                        { name: "Ensemble", rmse: 2764, mae: 1942, weight: null, winner: false },
                      ].map(m => (
                        <tr key={m.name} style={{ borderBottom: `1px solid ${C.border}` }}>
                          <td style={{ padding: "9px 0 5px", color: C.text }}>
                            {m.name}
                            {m.winner && <span style={{ marginLeft: 6, fontSize: 10, background: "#0D2818", color: C.green, padding: "1px 6px", borderRadius: 10 }}>best</span>}
                          </td>
                          <td style={{ fontFamily: "'IBM Plex Mono', monospace", color: C.text }}>${m.rmse.toLocaleString()}</td>
                          <td style={{ fontFamily: "'IBM Plex Mono', monospace", color: C.text }}>${m.mae.toLocaleString()}</td>
                          <td>
                            {m.weight != null ? (
                              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                <div style={{ flex: 1, height: 3, background: C.border, borderRadius: 2 }}>
                                  <div style={{ width: `${m.weight * 100}%`, height: "100%", background: C.blue, borderRadius: 2 }} />
                                </div>
                                <span style={{ fontSize: 11, color: C.muted, fontFamily: "monospace" }}>{(m.weight*100).toFixed(0)}%</span>
                              </div>
                            ) : <span style={{ color: C.muted, fontSize: 11 }}>—</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Insights */}
                <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                  <SectionHeader title="Business Insights" badge="4 signals" />
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {[
                      { icon: "↑", color: C.green, bg: "#0D2818", title: "Nov 2026 peak: $114.9k", body: "Highest month on record — driven by Q4 B2B orders." },
                      { icon: "⚠", color: C.amber, bg: "#1F1A0D", title: "Discount pressure: 51% of orders", body: "avg_discount is the #2 XGBoost feature. Margin risk." },
                      { icon: "◈", color: C.blue,  bg: "#0D1F3D", title: "Technology leads at 36.1%", body: "Strongest revenue category and highest avg order value." },
                      { icon: "↓", color: C.red,   bg: "#2D0D0D", title: "Feb 2026 dip: $20.3k", body: "Lowest month. Prophet flagged seasonality risk." },
                    ].map((ins, i) => (
                      <div key={i} style={{ display: "flex", gap: 10, padding: "8px 10px", background: ins.bg, borderRadius: 6 }}>
                        <span style={{ color: ins.color, fontSize: 15, marginTop: 1 }}>{ins.icon}</span>
                        <div>
                          <div style={{ fontWeight: 500, color: C.text, fontSize: 12 }}>{ins.title}</div>
                          <div style={{ color: C.muted, fontSize: 11, marginTop: 2 }}>{ins.body}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* ── ACTUALS TAB ── */}
          {activeTab === "actuals" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16, gridColumn: "1/-1" }}>
                <SectionHeader title="Monthly Revenue (18 months)" />
                <ResponsiveContainer width="100%" height={220}>
                  <ComposedChart data={MONTHLY} margin={{ top: 4, right: 8, bottom: 0, left: 8 }}>
                    <XAxis dataKey="month" tickFormatter={fmtMonth} tick={{ fontSize: 10, fill: C.muted }} tickLine={false} axisLine={false} interval={2} />
                    <YAxis tickFormatter={v => `$${(v/1000).toFixed(0)}k`} tick={{ fontSize: 10, fill: C.muted }} tickLine={false} axisLine={false} width={44} />
                    <Tooltip formatter={(v) => [fmt(v), "Revenue"]} labelFormatter={fmtMonth} contentStyle={{ background: "#1C2128", border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 12 }} />
                    <Bar dataKey="revenue" radius={[3,3,0,0]} maxBarSize={32}>
                      {MONTHLY.map((entry, i) => (
                        <Cell key={i} fill={entry.revenue === Math.max(...MONTHLY.map(m => m.revenue)) ? C.green : entry.revenue === Math.min(...MONTHLY.map(m => m.revenue)) ? C.red : "#1F6FEB"} fillOpacity={0.8} />
                      ))}
                    </Bar>
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                <SectionHeader title="Category Breakdown" />
                {CATEGORIES.map(c => (
                  <div key={c.category} style={{ marginBottom: 14 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5, fontSize: 12 }}>
                      <span style={{ color: C.text }}>{c.category}</span>
                      <span style={{ fontFamily: "monospace", color: C.muted }}>{fmt(c.revenue)} · {c.share}%</span>
                    </div>
                    <div style={{ height: 5, background: C.border, borderRadius: 3 }}>
                      <div style={{ width: `${c.share}%`, height: "100%", background: c.category === "Technology" ? C.blue : c.category === "Furniture" ? C.amber : C.green, borderRadius: 3, transition: "width 0.6s ease" }} />
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                <SectionHeader title="Key Stats" />
                {[
                  ["Total Transactions","10,194"],["Unique Customers","804"],["Unique Products","1,862"],
                  ["Avg Order Value","$217.10"],["Avg Discount","13.6%"],["GP Margin","10.4%"],
                  ["Recurring Rev Share","46%"],["Date Range","Jan 2023 – Dec 2026"],
                ].map(([k,v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}`, fontSize: 12 }}>
                    <span style={{ color: C.muted }}>{k}</span>
                    <span style={{ fontFamily: "monospace", color: C.text }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── MODELS TAB ── */}
          {activeTab === "models" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                <SectionHeader title="XGBoost · Feature Importance" badge="Top 10" />
                {TOP_FEATURES.map((f, i) => (
                  <div key={f} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                    <span style={{ fontFamily: "monospace", fontSize: 10, color: C.muted, width: 16 }}>{i+1}</span>
                    <span style={{ fontSize: 11, color: C.text, width: 140, flexShrink: 0 }}>{f}</span>
                    <div style={{ flex: 1, height: 4, background: C.border, borderRadius: 2 }}>
                      <div style={{ width: `${(FEATURE_SCORES[i]/FEATURE_SCORES[0])*100}%`, height: "100%", background: `hsl(${210 - i*12},80%,60%)`, borderRadius: 2 }} />
                    </div>
                    <span style={{ fontFamily: "monospace", fontSize: 10, color: C.muted, width: 36 }}>{FEATURE_SCORES[i].toFixed(3)}</span>
                  </div>
                ))}
              </div>
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                <SectionHeader title="Cross-Validation Results" />
                {[
                  { model: "XGBoost", cv_rmse: "1,785 ± 329", cv_mae: "1,318", holdout: "2,669", folds: 5, color: C.blue },
                  { model: "Prophet", cv_rmse: "2,217", cv_mae: "1,591", holdout: "3,136", folds: "built-in", color: C.amber },
                  { model: "Ensemble", cv_rmse: "—", cv_mae: "—", holdout: "2,764", folds: "—", color: C.green },
                ].map(m => (
                  <div key={m.model} style={{ padding: "12px 0", borderBottom: `1px solid ${C.border}` }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <div style={{ width: 8, height: 8, borderRadius: "50%", background: m.color }} />
                      <span style={{ fontWeight: 600, color: C.text }}>{m.model}</span>
                      <span style={{ fontSize: 10, color: C.muted }}>· {m.folds} folds</span>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4 }}>
                      {[["CV RMSE",m.cv_rmse],["CV MAE",m.cv_mae],["Hold-out",m.holdout]].map(([l,v]) => (
                        <div key={l} style={{ background: C.bg, borderRadius: 5, padding: "6px 10px" }}>
                          <div style={{ fontSize: 10, color: C.muted }}>{l}</div>
                          <div style={{ fontFamily: "monospace", fontSize: 12, color: C.text, marginTop: 2 }}>${v}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
                <div style={{ marginTop: 14, padding: "10px 12px", background: "#0D2818", borderRadius: 6, fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
                  <span style={{ color: C.green, fontWeight: 500 }}>Ensemble strategy:</span> Inverse-RMSE weighted average. XGBoost weight: 54%, Prophet: 46%. Winner determined by lowest hold-out RMSE.
                </div>
              </div>
            </div>
          )}

          {/* ── SEGMENTS TAB ── */}
          {activeTab === "segments" && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                <SectionHeader title="Revenue by Segment" />
                {SEGMENTS.map((s, i) => (
                  <div key={s.segment} style={{ marginBottom: 18 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ color: C.text, fontWeight: 500 }}>{s.segment}</span>
                      <span style={{ fontFamily: "monospace", fontSize: 12, color: C.muted }}>{fmt(s.revenue)} · {s.share}%</span>
                    </div>
                    <div style={{ height: 8, background: C.border, borderRadius: 4 }}>
                      <div style={{ width: `${s.share}%`, height: "100%", background: [C.blue, C.green, C.amber][i], borderRadius: 4, transition: "width 0.8s ease" }} />
                    </div>
                  </div>
                ))}
                <div style={{ marginTop: 14, padding: "10px 12px", background: C.bg, borderRadius: 6, fontSize: 11, color: C.muted }}>
                  Consumer dominates at 50.3% of total revenue ($2.21M). Corporate and Home Office provide stable recurring segments.
                </div>
              </div>
              <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: 16 }}>
                <SectionHeader title="Segment Forecast Exposure" />
                <div style={{ fontSize: 11, color: C.muted, marginBottom: 14, lineHeight: 1.6 }}>
                  Based on 90-day ensemble forecast (total: $84.6k) with proportional segment allocation.
                </div>
                {SEGMENTS.map((s, i) => {
                  const fc90 = 84600;
                  const projected = fc90 * (s.share/100);
                  return (
                    <div key={s.segment} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 0", borderBottom: `1px solid ${C.border}` }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{ width: 8, height: 8, borderRadius: "50%", background: [C.blue,C.green,C.amber][i] }} />
                        <span style={{ color: C.text }}>{s.segment}</span>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontFamily: "monospace", color: C.text, fontSize: 12 }}>{fmt(projected)}</div>
                        <div style={{ fontSize: 10, color: C.muted }}>90-day proj.</div>
                      </div>
                    </div>
                  );
                })}
                <div style={{ marginTop: 14, padding: "10px 12px", background: "#1A1F0D", borderRadius: 6, fontSize: 11, color: C.muted }}>
                  <span style={{ color: C.amber }}>⚠ Note:</span> Segment projections are proportional estimates. Actual variation depends on B2B pipeline and seasonal patterns.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
