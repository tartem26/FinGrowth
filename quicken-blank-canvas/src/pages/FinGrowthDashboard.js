import React, { useMemo, useState } from "react";
import {
  AppBar,
  Box,
  Button,
  Card,
  Chip,
  Container,
  Divider,
  Grid,
  IconButton,
  Stack,
  TextField,
  Toolbar,
  Typography,
  Menu,
  MenuItem,
} from "@mui/material";
import AddRoundedIcon from "@mui/icons-material/AddRounded";
import DeleteOutlineRoundedIcon from "@mui/icons-material/DeleteOutlineRounded";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import AutoFixHighRoundedIcon from "@mui/icons-material/AutoFixHighRounded";
import ArrowDropDownRoundedIcon from "@mui/icons-material/ArrowDropDownRounded";

// install recharts to have charts: npm i recharts
import {
  ResponsiveContainer,
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
} from "recharts";

// -----------------------------
// Design tokens
// -----------------------------
const TOKENS = {
  radius: 18,
  blur: 18,
  border: "1px solid rgba(15, 23, 42, 0.10)",
  shadow: "0 14px 40px rgba(2, 6, 23, 0.08)",
  glass: "rgba(255, 255, 255, 0.72)",
  glass2: "rgba(255, 255, 255, 0.60)",
};

const CATEGORIES_17 = [
  { key: "Income_Deposits", label: "Income" },
  { key: "Housing", label: "Housing" },
  { key: "Utilities_Telecom", label: "Utilities/Telecom" },
  { key: "Groceries_FoodAtHome", label: "Groceries" },
  { key: "Dining_FoodAway", label: "Dining" },
  { key: "Transportation_Variable", label: "Transport (var)" },
  { key: "Auto_Costs", label: "Auto Costs" },
  { key: "Healthcare_OOP", label: "Healthcare OOP" },
  { key: "Insurance_All", label: "Insurance (All)" },
  { key: "Debt_Payments", label: "Debt" },
  { key: "Savings_Investments", label: "Savings" },
  { key: "Education_Childcare", label: "Edu/Child" },
  { key: "Entertainment", label: "Entertainment" },
  { key: "Subscriptions_Memberships", label: "Subscriptions" },
  { key: "Cash_ATM_MiscTransfers", label: "Cash/ATM/Misc" },
  { key: "Pets", label: "Pets" },
  { key: "Travel", label: "Travel" },
];

const CLUSTER_META = [
  { name: "C1 - Low", range: "<$2.3k" },
  { name: "C2 - Lower Mid", range: "$2.3k - $4.2k" },
  { name: "C3 - Mid", range: "$4.2k - $6.2k" },
  { name: "C4 - Upper Mid", range: "$6.2k - $12.9k" },
  { name: "C5 - High", range: "$12.9k - $17.5k" },
  { name: "C6 - Top 5%", range: "≥$17.5k" },
];

const PREDICT_API_URL = "http://127.0.0.1:5055/predict";

// -----------------------------
// Autofill month generator
// -----------------------------
function mulberry32(seed) {
  // small deterministic rng so autofill feels stable
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function rBetween(rng, a, b) {
  return a + (b - a) * rng();
}

function capShares(shares, cap) {
  const s = { ...shares };
  let total = 0;
  for (const k of Object.keys(s)) total += Math.max(0, +s[k] || 0);

  if (total > cap && total > 0) {
    const scale = cap / total;
    for (const k of Object.keys(s)) s[k] = Math.max(0, (+s[k] || 0) * scale);
  }
  return s;
}

function buildAutofillMonth(clusterIdx, monthIndex) {
  const rng = mulberry32(42 + monthIndex * 97 + clusterIdx * 131);

  const INCOME_KEY = "Income_Deposits";

  // monthly income ranges
  const incomeRanges = [
    [1200, 2200],
    [2300, 4200],
    [4200, 6200],
    [6200, 12900],
    [12900, 17500],
    [17500, 40000],
  ];

  const [lo, hi] = incomeRanges[clusterIdx] ?? incomeRanges[3];
  const income = rBetween(rng, lo, hi);

  // start all outflows at 0 share
  const shares = {};
  for (const c of CATEGORIES_17) {
    if (c.key !== INCOME_KEY) shares[c.key] = 0.0;
  }

  // Convenience aliases
  const K = {
    Housing: "Housing",
    Utilities: "Utilities_Telecom",
    Groceries: "Groceries_FoodAtHome",
    Dining: "Dining_FoodAway",
    Transport: "Transportation_Variable",
    Auto: "Auto_Costs",
    Health: "Healthcare_OOP",
    Insurance: "Insurance_All",
    Debt: "Debt_Payments",
    Savings: "Savings_Investments",
    Edu: "Education_Childcare",
    Fun: "Entertainment",
    Subs: "Subscriptions_Memberships",
    Cash: "Cash_ATM_MiscTransfers",
    Pets: "Pets",
    Travel: "Travel",
  };

  // shape templates per cluster
  // cap = total share of outflows relative to income
  let cap = 0.86;

  if (clusterIdx === 0) {
    // low: essentials and debt heavy with little savings
    shares[K.Housing] = rBetween(rng, 0.36, 0.52);
    shares[K.Utilities] = rBetween(rng, 0.06, 0.11);
    shares[K.Groceries] = rBetween(rng, 0.11, 0.20);
    shares[K.Transport] = rBetween(rng, 0.05, 0.10);
    shares[K.Insurance] = rBetween(rng, 0.04, 0.08);
    shares[K.Health] = rBetween(rng, 0.01, 0.05);
    shares[K.Debt] = rBetween(rng, 0.10, 0.24);
    shares[K.Savings] = rBetween(rng, 0.00, 0.03);
    shares[K.Dining] = rBetween(rng, 0.01, 0.05);
    shares[K.Fun] = rBetween(rng, 0.00, 0.02);
    shares[K.Travel] = rBetween(rng, 0.00, 0.01);
    shares[K.Subs] = rBetween(rng, 0.00, 0.01);
    shares[K.Cash] = rBetween(rng, 0.00, 0.03);
    cap = rBetween(rng, 0.86, 0.95);
  } else if (clusterIdx === 1) {
    shares[K.Housing] = rBetween(rng, 0.30, 0.45);
    shares[K.Utilities] = rBetween(rng, 0.05, 0.10);
    shares[K.Groceries] = rBetween(rng, 0.09, 0.16);
    shares[K.Transport] = rBetween(rng, 0.05, 0.10);
    shares[K.Insurance] = rBetween(rng, 0.04, 0.08);
    shares[K.Health] = rBetween(rng, 0.01, 0.05);
    shares[K.Debt] = rBetween(rng, 0.08, 0.18);
    shares[K.Savings] = rBetween(rng, 0.02, 0.08);
    shares[K.Dining] = rBetween(rng, 0.02, 0.07);
    shares[K.Fun] = rBetween(rng, 0.00, 0.03);
    shares[K.Travel] = rBetween(rng, 0.00, 0.02);
    shares[K.Subs] = rBetween(rng, 0.00, 0.02);
    shares[K.Cash] = rBetween(rng, 0.01, 0.04);
    cap = rBetween(rng, 0.82, 0.93);
  } else if (clusterIdx === 2) {
    shares[K.Housing] = rBetween(rng, 0.24, 0.36);
    shares[K.Utilities] = rBetween(rng, 0.04, 0.08);
    shares[K.Groceries] = rBetween(rng, 0.07, 0.13);
    shares[K.Transport] = rBetween(rng, 0.04, 0.09);
    shares[K.Insurance] = rBetween(rng, 0.03, 0.07);
    shares[K.Health] = rBetween(rng, 0.01, 0.05);
    shares[K.Debt] = rBetween(rng, 0.04, 0.12);
    shares[K.Savings] = rBetween(rng, 0.08, 0.18);
    shares[K.Dining] = rBetween(rng, 0.03, 0.09);
    shares[K.Fun] = rBetween(rng, 0.01, 0.04);
    shares[K.Travel] = rBetween(rng, 0.00, 0.04);
    shares[K.Subs] = rBetween(rng, 0.005, 0.02);
    shares[K.Cash] = rBetween(rng, 0.01, 0.05);
    cap = rBetween(rng, 0.78, 0.90);
  } else if (clusterIdx === 3) {
    shares[K.Housing] = rBetween(rng, 0.20, 0.32);
    shares[K.Utilities] = rBetween(rng, 0.03, 0.07);
    shares[K.Groceries] = rBetween(rng, 0.06, 0.11);
    shares[K.Transport] = rBetween(rng, 0.03, 0.08);
    shares[K.Insurance] = rBetween(rng, 0.03, 0.06);
    shares[K.Health] = rBetween(rng, 0.01, 0.04);
    shares[K.Debt] = rBetween(rng, 0.02, 0.08);
    shares[K.Savings] = rBetween(rng, 0.14, 0.30);
    shares[K.Dining] = rBetween(rng, 0.03, 0.10);
    shares[K.Fun] = rBetween(rng, 0.01, 0.05);
    shares[K.Travel] = rBetween(rng, 0.01, 0.06);
    shares[K.Subs] = rBetween(rng, 0.01, 0.03);
    shares[K.Cash] = rBetween(rng, 0.01, 0.05);
    cap = rBetween(rng, 0.72, 0.88);
  } else if (clusterIdx === 4) {
    shares[K.Housing] = rBetween(rng, 0.14, 0.26);
    shares[K.Utilities] = rBetween(rng, 0.02, 0.06);
    shares[K.Groceries] = rBetween(rng, 0.04, 0.09);
    shares[K.Transport] = rBetween(rng, 0.02, 0.06);
    shares[K.Insurance] = rBetween(rng, 0.02, 0.05);
    shares[K.Health] = rBetween(rng, 0.01, 0.03);
    shares[K.Debt] = rBetween(rng, 0.00, 0.05);
    shares[K.Savings] = rBetween(rng, 0.20, 0.36);
    shares[K.Dining] = rBetween(rng, 0.04, 0.10);
    shares[K.Fun] = rBetween(rng, 0.02, 0.06);
    shares[K.Travel] = rBetween(rng, 0.02, 0.08);
    shares[K.Subs] = rBetween(rng, 0.01, 0.04);
    shares[K.Cash] = rBetween(rng, 0.01, 0.06);
    cap = rBetween(rng, 0.65, 0.85);
  } else {
    // top: high savings and more discretionary
    shares[K.Housing] = rBetween(rng, 0.08, 0.18);
    shares[K.Utilities] = rBetween(rng, 0.01, 0.04);
    shares[K.Groceries] = rBetween(rng, 0.03, 0.08);
    shares[K.Transport] = rBetween(rng, 0.02, 0.06);
    shares[K.Auto] = rBetween(rng, 0.01, 0.05);      // more likely to have car payment/maintenance
    shares[K.Insurance] = rBetween(rng, 0.02, 0.05);
    shares[K.Health] = rBetween(rng, 0.01, 0.03);
    shares[K.Debt] = rBetween(rng, 0.00, 0.03);
    shares[K.Savings] = rBetween(rng, 0.25, 0.48);
    shares[K.Dining] = rBetween(rng, 0.04, 0.12);
    shares[K.Fun] = rBetween(rng, 0.02, 0.08);
    shares[K.Travel] = rBetween(rng, 0.03, 0.12);
    shares[K.Subs] = rBetween(rng, 0.02, 0.06);
    shares[K.Cash] = rBetween(rng, 0.01, 0.06);
    cap = rBetween(rng, 0.55, 0.82);
  }

  // light fill so remaining categories aren't empty
  for (const k of [
    K.Transport,
    K.Auto,
    K.Health,
    K.Insurance,
    K.Edu,
    K.Pets,
    K.Cash,
  ]) {
    if (k in shares && (shares[k] || 0) === 0) {
      shares[k] = rBetween(rng, 0.0, 0.06);
    }
  }

  const capped = capShares(shares, cap);

  // output as strings
  const out = {};
  out[INCOME_KEY] = String(Math.round(income));

  for (const c of CATEGORIES_17) {
    if (c.key === INCOME_KEY) continue;
    out[c.key] = String(Math.round((capped[c.key] || 0) * income));
  }
  return out;
}

// -----------------------------
// Glass card
// -----------------------------
function GlassCard({ title, subtitle, actions, children, height = 280 }) {
  return (
    <Card
      elevation={0}
      sx={{
        position: "relative",
        height,
        overflow: "hidden",
        borderRadius: TOKENS.radius,
        border: TOKENS.border,
        background: `linear-gradient(180deg, ${TOKENS.glass}, ${TOKENS.glass2})`,
        backdropFilter: `blur(${TOKENS.blur}px)`,
        boxShadow: TOKENS.shadow,
        padding: 0.5,
      }}
    >
      {/* Blue gradient triangle */}
      <Box
        sx={{
          position: "absolute",
          top: -90,
          right: -90,
          width: 260,
          height: 260,
          background: "linear-gradient(135deg, rgba(59,130,246,0.90), rgba(147,197,253,0.15))",
          clipPath: "polygon(0 0, 100% 0, 100% 100%)",
          opacity: 0.95,
        }}
      />

      {/* important: flex column so charts never overlap */}
      <Box
        sx={{
          position: "relative",
          height: "100%",
          p: { xs: 2.2, md: 2.8 },
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Stack direction="row" alignItems="flex-start" justifyContent="space-between" spacing={2}>
          <Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 800, letterSpacing: 0.2, lineHeight: 1.15 }}>
              {title}
            </Typography>
            {subtitle ? (
              <Typography variant="caption" sx={{ opacity: 0.75, display: "block", mt: 0.4 }}>
                {subtitle}
              </Typography>
            ) : null}
          </Box>
          {actions ? <Box>{actions}</Box> : null}
        </Stack>

        <Divider sx={{ mt: 1.4, mb: 2.0, opacity: 0.25 }} />

        {/* important: minHeight 0 so ResponsiveContainer can measure */}
        <Box sx={{ flex: 1, minHeight: 0 }}>{children}</Box>
      </Box>
    </Card>
  );
}

function Pill({ label, value }) {
  return (
    <Box
      sx={{
        px: 1.1,
        py: 0.7,
        borderRadius: 999,
        border: "1px solid rgba(15, 23, 42, 0.10)",
        background: "rgba(255,255,255,0.62)",
      }}
    >
      <Stack direction="row" spacing={1} alignItems="center">
        <Typography variant="caption" sx={{ opacity: 0.7 }}>
          {label}
        </Typography>
        <Typography variant="caption" sx={{ fontWeight: 800 }}>
          {value}
        </Typography>
      </Stack>
    </Box>
  );
}

// -----------------------------
// Main page
// -----------------------------
export default function FinGrowthDashboard() {
  const [rows, setRows] = useState(() => [emptyRow()]);
  const [savedAt, setSavedAt] = useState(null);
  const [clusterResult, setClusterResult] = useState(() => ({
    top: 0,
    probs: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
  }));
  const [apiBusy, setApiBusy] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [apiError, setApiError] = useState(null);

  const radar1 = analysis?.radars?.snapshot ?? [];
  const radar2 = analysis?.radars?.trend ?? [];
  const radar3 = analysis?.radars?.risk ?? [];
  const radar4 = analysis?.radars?.growth ?? [];

  const networth = analysis?.networth ?? [];              // [{month, networth}, ...]
  const scatter = analysis?.cluster_space?.points ?? [];  // [{x,y,k,isUser?}, ...]
  const conclusion = analysis?.conclusion ?? null;
  const warnings = analysis?.warnings ?? [];

  function emptyRow() {
    const r = {};
    for (const c of CATEGORIES_17) r[c.key] = "";
    return r;
  }

  function addRow() {
    setRows((prev) => [...prev, emptyRow()]);
  }

  function deleteRow(i) {
    setRows((prev) => (prev.length <= 1 ? prev : prev.filter((_, idx) => idx !== i)));
  }

  function updateCell(rowIdx, key, value) {
    setRows((prev) => {
      const next = [...prev];
      next[rowIdx] = { ...next[rowIdx], [key]: value };
      return next;
    });
  }

  function autofillRow(rowIdx, clusterIdx) {
    const filled = buildAutofillMonth(clusterIdx, rowIdx);

    setRows((prev) => {
      const next = [...prev];
      next[rowIdx] = { ...next[rowIdx], ...filled };
      return next;
    });
  }

  async function onSave() {
    setApiBusy(true);
    setApiError(null);

    try {
      const res = await fetch(PREDICT_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ months: rows }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`API error ${res.status}: ${txt}`);
      }

      const out = await res.json();

      // Must include top/probs at minimum
      if (!out || typeof out.top !== "number" || !Array.isArray(out.probs)) {
        throw new Error("Bad API response shape");
      }

      // --- Normalize backend output into the exact shape as UI expects ---
      // Backend returns: radars = { snapshot, trend, risk, growth }
      // UI expects: analysis.radars.snapshot/trend/risk/growth
      const normalizedRadars = {
        snapshot: out.radars?.snapshot ?? [],
        trend: out.radars?.trend ?? [],
        risk: out.radars?.risk ?? [],
        growth: out.radars?.growth ?? [],
      };

      // Backend returns: cluster_space = { points: [...], user_point: {...} }
      // IMPORTANT: Backend may already include the user point inside points[] with isUser=true.
      const pts = Array.isArray(out.cluster_space?.points) ? out.cluster_space.points : [];
      const hasUserInPts = pts.some((p) => p && p.isUser);

      const userPt =
        !hasUserInPts && out.cluster_space?.user_point
          ? [{ ...out.cluster_space.user_point, isUser: true }]
          : [];

      const normalizedClusterSpace = {
        points: [...pts, ...userPt],
      };

      const normalized = {
        ...out,
        radars: normalizedRadars,
        networth: Array.isArray(out.networth) ? out.networth : [],
        cluster_space: normalizedClusterSpace,
        conclusion: out.conclusion ?? null,
        warnings: Array.isArray(out.warnings) ? out.warnings : [],
      };

      setClusterResult({ top: out.top, probs: out.probs });
      setAnalysis(normalized); // charts come from backend only
      setSavedAt(new Date());
    } catch (e) {
      console.error(e);
      setApiError(e.message || String(e));
    } finally {
      setApiBusy(false);
    }
  }

  const topMeta = CLUSTER_META[clusterResult.top];

  return (
    <Box sx={{ minHeight: "100vh", background: "#ffffff" }}>
      <AppBar
        elevation={0}
        position="sticky"
        sx={{
          background: "rgba(255,255,255,0.72)",
          backdropFilter: `blur(${TOKENS.blur}px)`,
          borderBottom: "1px solid rgba(15, 23, 42, 0.08)",
        }}
      >
        <Toolbar>
          <Container maxWidth="xl" disableGutters sx={{ px: { xs: 2, md: 3 } }}>
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Stack direction="row" alignItems="center" spacing={1.2}>
                <Box
                  sx={{
                    width: 38,
                    height: 38,
                    borderRadius: 12,
                    background: "linear-gradient(135deg, rgba(59,130,246,1), rgba(147,197,253,0.35))",
                    border: "1px solid rgba(15, 23, 42, 0.10)",
                    boxShadow: "0 10px 22px rgba(59,130,246,0.18)",
                  }}
                />
                <Box>
                  <Typography sx={{ fontWeight: 900, letterSpacing: 0.2, color: "#0f172a" }}>
                    FinGrowth
                  </Typography>
                  <Typography variant="caption" sx={{ opacity: 0.7, color: "#0f172a" }}>
                    Personal finance trajectory dashboard
                  </Typography>
                </Box>
              </Stack>

              <Stack direction="row" spacing={1} alignItems="center">
                <Chip
                  label={`${rows.length} month${rows.length === 1 ? "" : "s"} added`}
                  sx={{
                    borderRadius: 999,
                    background: "rgba(59,130,246,0.08)",
                    border: "1px solid rgba(59,130,246,0.18)",
                    fontWeight: 700,
                  }}
                />
                <Chip
                  label={savedAt ? `Updated ${savedAt.toLocaleTimeString()}` : "Not saved"}
                  sx={{
                    borderRadius: 999,
                    background: "rgba(2,6,23,0.04)",
                    border: "1px solid rgba(15,23,42,0.10)",
                    fontWeight: 700,
                  }}
                />
              </Stack>
            </Stack>
          </Container>
        </Toolbar>
      </AppBar>

      {/* make the whole dashboard row-width bigger, so charts can grow */}
      <Container  sx={{ py: { xs: 3.2, md: 4.2 }, px: { xs: 2, md: 3, lg: 6 } }}>
        {/* strict layout: 4 rows only, no mixing */}
        <Stack spacing={{ xs: 2.4, md: 3.0 }} alignItems="center">
          {/* Row 1: ONLY 4 radars, full width */}
          <Grid container spacing={{ xs: 2.2, md: 2.6 }} sx={{ width: "100%" }}>
            <Grid item xs={12} sm={6} lg={3} sx={{ width: "23%", alignSelf: "stretch" }}>
              <GlassCard title="Radar · Snapshot" subtitle="Last saved month profile" height={360}>
                <RadarPanel data={radar1} />
              </GlassCard>
            </Grid>
            <Grid item xs={12} sm={6} lg={3} sx={{ width: "23%", alignSelf: "stretch" }}>
              <GlassCard title="Radar · Trend" subtitle="Rolling behavior signals" height={360}>
                <RadarPanel data={radar2} />
              </GlassCard>
            </Grid>
            <Grid item xs={12} sm={6} lg={3} sx={{ width: "23%", alignSelf: "stretch" }}>
              <GlassCard title="Radar · Risk" subtitle="Debt & essentials pressure" height={360}>
                <RadarPanel data={radar3} />
              </GlassCard>
            </Grid>
            <Grid item xs={12} sm={6} lg={3} sx={{ width: "23%", alignSelf: "stretch" }}>
              <GlassCard title="Radar · Growth" subtitle="Savings momentum" height={360}>
                <RadarPanel data={radar4} />
              </GlassCard>
            </Grid>
          </Grid>

          {/* Row 2: ONLY net worth (left) and clusters (right), full width */}
          <Grid container spacing={{ xs: 2.2, md: 2.6 }} sx={{ width: "100%" }}>
            <Grid item xs={12} md={6} sx={{ width: "49%", alignSelf: "stretch" }}>
              <GlassCard title="Net worth" subtitle="Savings / net worth projection" height={460}>
                <NetworthPanel data={networth} />
              </GlassCard>
            </Grid>

            <Grid item xs={12} md={6} sx={{ width: "49%", alignSelf: "stretch" }}>
              <GlassCard
                title="Cluster space"
                subtitle="All 6 clusters overview"
                height={460}
                width={760}
                actions={
                  <Stack direction="row" spacing={1}>
                    <Pill label="Top" value={topMeta.name} />
                    <Pill label="Range" value={topMeta.range} />
                  </Stack>
                }
              >
                <ClusterScatterPanel data={scatter} />
              </GlassCard>
            </Grid>
          </Grid>

          {/* Row 3: ONLY monthly inputs, full width */}
          <GlassCard
            title="Monthly inputs"
            subtitle="Add 1+ months. More months → better trajectory estimation."
            height={rows.length <= 2 ? 620 : 720}
            actions={
              <Button
                variant="contained"
                onClick={onSave}
                disabled={apiBusy}
                startIcon={<CheckCircleRoundedIcon />}
                sx={{
                  borderRadius: 999,
                  textTransform: "none",
                  fontWeight: 900,
                  background: "linear-gradient(135deg, rgba(34,197,94,1), rgba(16,185,129,0.85))",
                  boxShadow: "0 14px 30px rgba(34,197,94,0.22)",
                  px: 2.2,
                  py: 1.1,
                }}
              >
                Save
              </Button>
            }
          >
            <InputsPanel rows={rows} onAdd={addRow} onDelete={deleteRow} onChange={updateCell} onAutofill={autofillRow} />
          </GlassCard>

          {/* Row 4: ONLY conclusion */}
          <ResultsPanel
            clusterResult={clusterResult}
            conclusion={conclusion}
            warnings={warnings}
            apiError={apiError}
          />
        </Stack>
      </Container>

      <Box sx={{ py: 3, opacity: 0.75 }}>
        <Container maxWidth={false} sx={{ px: { xs: 2, md: 3, lg: 6 } }}>
          <Typography variant="caption">FinGrowth 2026 · ML pipeline by Artem Tikhonov.</Typography>
        </Container>
      </Box>
    </Box>
  );
}

// -----------------------------
// Charts
// -----------------------------
function RadarPanel({ data }) {
  if (!data || data.length === 0) {
    return (
      <Box sx={{ height: "100%", display: "grid", placeItems: "center", opacity: 0.7 }}>
        <Typography variant="caption">Save to compute radar chart</Typography>
      </Box>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart data={data} outerRadius="74.5%">
        <PolarGrid strokeOpacity={0.25} />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
        <Radar dataKey="v" stroke="rgba(59,130,246,0.9)" fill="rgba(59,130,246,0.30)" />
        <Tooltip />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function ClusterScatterPanel({ data }) {
  const userPts = useMemo(() => data.filter((p) => p.isUser), [data]);
  const cloudPts = useMemo(() => data.filter((p) => !p.isUser), [data]);

  const grouped = useMemo(() => {
    const g = new Map();
    for (const p of cloudPts) {
      if (!g.has(p.k)) g.set(p.k, []);
      g.get(p.k).push(p);
    }
    return [...g.entries()].sort((a, b) => a[0] - b[0]);
  }, [cloudPts]);

  if (!data || data.length === 0) {
    return (
      <Box sx={{ height: "100%", display: "grid", placeItems: "center", opacity: 0.7 }}>
        <Typography variant="caption">Save to compute cluster space</Typography>
      </Box>
    );
  }

  const palette = [
    "rgba(59,130,246,0.85)",
    "rgba(14,165,233,0.85)",
    "rgba(99,102,241,0.85)",
    "rgba(34,197,94,0.80)",
    "rgba(245,158,11,0.85)",
    "rgba(244,63,94,0.80)",
  ];

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 8, right: 14, left: 0, bottom: 6 }}>
        <CartesianGrid strokeOpacity={0.25} />
        <XAxis type="number" dataKey="x" tick={{ fontSize: 12 }} />
        <YAxis type="number" dataKey="y" tick={{ fontSize: 12 }} />
        <Tooltip cursor={{ strokeOpacity: 0.2 }} />
        <Legend />

        {grouped
          .filter(([k]) => k >= 0)
          .map(([k, pts]) => (
            <Scatter
              key={k}
              name={CLUSTER_META[k]?.name ?? `C${k + 1}`} // optional: 1-based label
              data={pts}
              fill={palette[k % palette.length]}
            />
          ))
        }


        {userPts.length ? (
          <Scatter name="You" data={userPts} fill="rgba(2,6,23,0.95)" />
        ) : null}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function NetworthPanel({ data }) {
  if (!data || data.length === 0) {
    return (
      <Box sx={{ height: "100%", display: "grid", placeItems: "center", opacity: 0.7 }}>
        <Typography variant="caption">Save to compute net worth chart</Typography>
      </Box>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 6 }}>
        <CartesianGrid strokeOpacity={0.25} />
        <XAxis dataKey="month" tick={{ fontSize: 12 }} />
        <YAxis tick={{ fontSize: 12 }} width={46} />
        <Tooltip />
        <Line type="monotone" dataKey="networth" stroke="rgba(59,130,246,0.9)" strokeWidth={3} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// -----------------------------
// Inputs panel
// -----------------------------
function InputsPanel({ rows, onAdd, onDelete, onChange, onAutofill }) {
  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column", minHeight: 0 }}>
      <Box sx={{ flex: 1, minHeight: 0, overflow: "auto", pr: 0.6 }}>
        <Stack spacing={2.0}>
          {rows.map((r, idx) => (
            <MonthRow
              key={idx}
              index={idx}
              row={r}
              canDelete={rows.length > 1}
              onDelete={() => onDelete(idx)}
              onChange={(key, value) => onChange(idx, key, value)}
              onAutofill={(clusterIdx) => onAutofill(idx, clusterIdx)}
            />
          ))}
        </Stack>
      </Box>

      <Stack direction="row" alignItems="center" justifyContent="flex-end" sx={{ pt: 2.0 }}>
        <Button
          onClick={onAdd}
          startIcon={<AddRoundedIcon />}
          variant="contained"
          sx={{
            borderRadius: 999,
            textTransform: "none",
            fontWeight: 900,
            background: "linear-gradient(135deg, rgba(37,99,235,1), rgba(96,165,250,0.85))",
            boxShadow: "0 14px 28px rgba(59,130,246,0.18)",
            px: 2.2,
            py: 1.1,
          }}
        >
          Add month
        </Button>
      </Stack>
    </Box>
  );
}

function MonthRow({ index, row, onChange, onDelete, canDelete, onAutofill }) {
    const [autoAnchor, setAutoAnchor] = useState(null);
    const autoOpen = Boolean(autoAnchor);

    function openAutofillMenu(e) {
        setAutoAnchor(e.currentTarget);
    }

    function closeAutofillMenu() {
        setAutoAnchor(null);
    }

    function pickAutofill(i) {
        closeAutofillMenu();
        onAutofill(i);
    }

    return (
    <Box
      sx={{
        borderRadius: TOKENS.radius,
        border: "1px solid rgba(15, 23, 42, 0.10)",
        background: "rgba(255,255,255,0.70)",
        backdropFilter: `blur(${TOKENS.blur}px)`,
        p: { xs: 1.4, md: 1.8 },
      }}
    >
      <Stack spacing={1.4} padding={0.7}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Stack direction="row" spacing={1} alignItems="center">
            <Chip
              label={`Month ${index + 1}`}
              size="small"
              sx={{
                borderRadius: 999,
                fontWeight: 800,
                background: "rgba(2,6,23,0.04)",
                border: "1px solid rgba(15,23,42,0.10)",
              }}
            />
            <Typography variant="caption" sx={{ opacity: 0.75 }}>
              Enter monthly totals (USD)
            </Typography>
          </Stack>

          <Stack direction="row" spacing={1} alignItems="center">
            <Button
                onClick={openAutofillMenu}
                startIcon={<AutoFixHighRoundedIcon />}
                endIcon={<ArrowDropDownRoundedIcon />}
                variant="contained"
                size="small"
                sx={{
                borderRadius: 999,
                textTransform: "none",
                fontWeight: 900,
                background: "rgba(59,130,246,0.10)",
                color: "rgba(15,23,42,0.92)",
                border: "1px solid rgba(59,130,246,0.22)",
                boxShadow: "none",
                "&:hover": { background: "rgba(59,130,246,0.14)", boxShadow: "none" },
                }}
            >
                Autofill
            </Button>

            <Menu
                anchorEl={autoAnchor}
                open={autoOpen}
                onClose={closeAutofillMenu}
                PaperProps={{
                sx: {
                    borderRadius: 8,
                    border: "1px solid rgba(15, 23, 42, 0.10)",
                    boxShadow: TOKENS.shadow,
                    overflow: "hidden",
                },
                }}
            >
                {CLUSTER_META.map((c, i) => (
                <MenuItem key={c.name} onClick={() => pickAutofill(i)}>
                    <Box sx={{ width: "100%" }}>
                    <Typography sx={{ fontWeight: 900, fontSize: 13 }}>{c.name}</Typography>
                    <Typography variant="caption" sx={{ opacity: 0.7 }}>
                        {c.range}
                    </Typography>
                    </Box>
                </MenuItem>
                ))}
            </Menu>

            <IconButton
                onClick={onDelete}
                disabled={!canDelete}
                sx={{
                width: 42,
                height: 42,
                borderRadius: 12,
                border: "1px solid rgba(244,63,94,0.25)",
                background: canDelete ? "rgba(244,63,94,0.10)" : "rgba(2,6,23,0.03)",
                }}
            >
                <DeleteOutlineRoundedIcon sx={{ color: canDelete ? "rgba(244,63,94,0.9)" : "rgba(2,6,23,0.35)" }} />
            </IconButton>
          </Stack>
        </Stack>

        <Grid container spacing={{ xs: 1.2, md: 1.4 }}>
          {CATEGORIES_17.map((c) => (
            <Grid item xs={6} sm={4} md={3} lg={2} key={c.key}>
              <TextField
                label={c.label}
                value={row[c.key]}
                onChange={(e) => onChange(c.key, e.target.value)}
                size="small"
                fullWidth
                placeholder="0"
                inputProps={{ inputMode: "decimal" }}
                sx={{
                  "& .MuiOutlinedInput-root": {
                    borderRadius: 14,
                    background: "rgba(255,255,255,0.70)",
                  },
                  "& label": { fontWeight: 700 },
                }}
              />
            </Grid>
          ))}
        </Grid>
      </Stack>
    </Box>
  );
}

// -----------------------------
// Results panel
// -----------------------------
function ResultsPanel({ clusterResult, conclusion, warnings, apiError }) {
  const top = clusterResult?.top ?? 0;
  const probs = Array.isArray(clusterResult?.probs) ? clusterResult.probs : [];

  const topProb = probs.length ? Math.max(...probs) : 0;

  const drivers = Array.isArray(conclusion?.drivers) ? conclusion.drivers : [];
  const missing = Array.isArray(conclusion?.missing_fields) ? conclusion.missing_fields : [];

  return (
    <Card
      elevation={0}
      sx={{
        borderRadius: TOKENS.radius,
        border: TOKENS.border,
        background: "rgba(255,255,255,0.72)",
        backdropFilter: `blur(${TOKENS.blur}px)`,
        boxShadow: TOKENS.shadow,
      }}
    >
      <Box sx={{ p: { xs: 2.2, md: 2.8 } }}>
        <Stack spacing={1.6}>
          <Stack
            direction={{ xs: "column", md: "row" }}
            alignItems={{ xs: "flex-start", md: "center" }}
            justifyContent="space-between"
            spacing={1.4}
            padding={0.5}
          >
            <Box>
              <Typography sx={{ fontWeight: 900, fontSize: 18 }}>Conclusion</Typography>
              <Typography variant="caption" sx={{ opacity: 0.75 }}>
                After Save, FinGrowth shows the predicted financial clusters with their probabilistic breakdown.
              </Typography>

              {/* API error (red) */}
              {apiError ? (
                <Box
                  sx={{
                    mt: 1.2,
                    p: 1.2,
                    borderRadius: 5,
                    border: "1px solid rgba(244,63,94,0.35)",
                    background: "rgba(244,63,94,0.08)",
                  }}
                >
                  <Typography sx={{ color: "rgba(190,18,60,0.98)", fontWeight: 900, fontSize: 13 }}>
                    Error
                  </Typography>
                  <Typography sx={{ mt: 0.4, color: "rgba(190,18,60,0.95)", fontWeight: 800 }}>
                    {apiError}
                  </Typography>
                </Box>
              ) : null}

              {/* Server warnings (muted yellow) */}
              {Array.isArray(warnings) && warnings.length ? (
                <Box
                  sx={{
                    mt: 1.0,
                    p: 1.2,
                    borderRadius: 5,
                    border: "1px solid rgba(245,158,11,0.35)",
                    background: "rgba(245,158,11,0.10)",
                  }}
                >
                  <Typography sx={{ color: "rgba(146,64,14,0.98)", fontWeight: 900, fontSize: 13 }}>
                    Warnings
                  </Typography>
                  <Box sx={{ mt: 0.6 }}>
                    {warnings.map((w, i) => (
                      <Typography key={i} variant="caption" sx={{ display: "block", color: "rgba(146,64,14,0.92)" }}>
                        • {w}
                      </Typography>
                    ))}
                  </Box>
                </Box>
              ) : null}

              {/* Server-side conclusion text */}
              {conclusion?.text ? (
                <Typography sx={{ mt: 1.1, opacity: 0.92 }}>{conclusion.text}</Typography>
              ) : null}
            </Box>

            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
              <Chip
                label={`${CLUSTER_META[top]?.name ?? "Unknown"} · ${CLUSTER_META[top]?.range ?? ""}`}
                sx={{
                  borderRadius: 999,
                  fontWeight: 900,
                  background: "rgba(34,197,94,0.10)",
                  border: "1px solid rgba(34,197,94,0.25)",
                }}
              />
              <Chip
                label={`Top prob: ${(topProb * 100).toFixed(1)}%`}
                sx={{
                  borderRadius: 999,
                  fontWeight: 900,
                  background: "rgba(59,130,246,0.08)",
                  border: "1px solid rgba(59,130,246,0.18)",
                }}
              />
            </Stack>
          </Stack>

          <Divider sx={{ opacity: 0.25 }} />

          {/* Drivers section: between conclusion text and probability tiles */}
          {(drivers.length || missing.length) ? (
            <Box
              sx={{
                borderRadius: 8,
                border: "1px solid rgba(15, 23, 42, 0.10)",
                background: "rgba(255,255,255,0.70)",
                p: { xs: 1.6, md: 1.8 },
              }}
            >
              <Stack spacing={1.0}>
                {drivers.length ? (
                  <Box>
                    <Typography sx={{ fontWeight: 900, fontSize: 14 }}>Top drivers</Typography>
                    <Box sx={{ mt: 0.6 }}>
                      {drivers.map((d, i) => (
                        <Typography key={i} variant="caption" sx={{ display: "block", opacity: 0.82 }}>
                          • {d}
                        </Typography>
                      ))}
                    </Box>
                  </Box>
                ) : null}

                {missing.length ? (
                  <Box>
                    <Typography sx={{ fontWeight: 900, fontSize: 14, mt: drivers.length ? 0.6 : 0 }}>
                      Missing fields
                    </Typography>
                    <Box sx={{ mt: 0.6 }}>
                      {missing.slice(0, 6).map((m, i) => (
                        <Typography key={i} variant="caption" sx={{ display: "block", opacity: 0.78 }}>
                          • {m}
                        </Typography>
                      ))}
                      {missing.length > 6 ? (
                        <Typography variant="caption" sx={{ display: "block", opacity: 0.65 }}>
                          +{missing.length - 6} more
                        </Typography>
                      ) : null}
                    </Box>
                  </Box>
                ) : null}
              </Stack>
            </Box>
          ) : null}

          {probs.length ? (
            <Grid container spacing={{ xs: 1.2, md: 1.6 }}>
              {probs.map((p, i) => (
                <Grid item xs={12} sm={6} md={4} lg={2} key={i}>
                  <Box
                    sx={{
                      borderRadius: 18,
                      border: "1px solid rgba(15, 23, 42, 0.10)",
                      background: "rgba(255,255,255,0.70)",
                      p: 1.8,
                      position: "relative",
                      overflow: "hidden",
                      minHeight: 110,
                      minWidth: 165,
                    }}
                  >
                    <Box
                      sx={{
                        position: "absolute",
                        inset: 0,
                        background:
                          i === top
                            ? "linear-gradient(135deg, rgba(34,197,94,0.16), rgba(34,197,94,0.02))"
                            : "linear-gradient(135deg, rgba(59,130,246,0.12), rgba(59,130,246,0.01))",
                        pointerEvents: "none",
                      }}
                    />
                    <Stack spacing={0.6} sx={{ position: "relative" }} alignItems="center">
                      <Typography variant="caption" sx={{ opacity: 0.75, fontWeight: 800 }}>
                        {CLUSTER_META[i]?.name ?? `C${i + 1}`}
                      </Typography>
                      <Typography sx={{ fontWeight: 1000, fontSize: 24, lineHeight: 1.05 }}>
                        {(Number(p) * 100).toFixed(2)}%
                      </Typography>
                      <Typography variant="caption" sx={{ opacity: 0.65 }}>
                        {CLUSTER_META[i]?.range ?? ""}
                      </Typography>
                    </Stack>
                  </Box>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Box sx={{ py: 2.2, opacity: 0.75 }}>
              <Typography variant="caption">
                Save to compute probabilities and generate a model-backed conclusion.
              </Typography>
            </Box>
          )}
        </Stack>
      </Box>
    </Card>
  );
}
