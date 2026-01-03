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
} from "@mui/material";
import AddRoundedIcon from "@mui/icons-material/AddRounded";
import DeleteOutlineRoundedIcon from "@mui/icons-material/DeleteOutlineRounded";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";

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
  { key: "Utilities_Telecom", label: "Utilities" },
  { key: "Groceries_FoodAtHome", label: "Groceries" },
  { key: "Dining_FoodAway", label: "Dining" },
  { key: "Transportation_Gas", label: "Gas" },
  { key: "Transportation_PublicTransit", label: "Transit" },
  { key: "Insurance_Health", label: "Health Ins" },
  { key: "Insurance_Auto", label: "Auto Ins" },
  { key: "Medical_OutOfPocket", label: "Medical" },
  { key: "Debt_Payments", label: "Debt" },
  { key: "Savings_Investments", label: "Savings" },
  { key: "Education_Childcare", label: "Edu/Child" },
  { key: "Entertainment", label: "Fun" },
  { key: "Subscriptions_Memberships", label: "Subs" },
  { key: "Pets", label: "Pets" },
  { key: "Travel", label: "Travel" },
];

const CLUSTER_META = [
  { name: "C1_low", range: "<$2.3k" },
  { name: "C2_lower_mid", range: "$2.3k–$4.2k" },
  { name: "C3_mid", range: "$4.2k–$6.2k" },
  { name: "C4_upper_mid", range: "$6.2k–$12.9k" },
  { name: "C5_high", range: "$12.9k–$17.5k" },
  { name: "C6_top5", range: ">=$17.5k" },
];

// -----------------------------
// Mock chart data (UI-only)
// -----------------------------
function makeRadarData(seed = 0) {
  const base = [
    { metric: "Essentials", v: 62 },
    { metric: "Debt", v: 35 },
    { metric: "Savings", v: 55 },
    { metric: "Discretionary", v: 40 },
    { metric: "Net Flow", v: 58 },
  ];
  return base.map((d, i) => ({
    ...d,
    v: Math.max(10, Math.min(95, d.v + ((i + seed) % 3 === 0 ? 8 : -6) + (seed % 5))),
  }));
}

function makeClusterScatter() {
  // a simple 2D mock with 6 clusters
  const pts = [];
  const centers = [
    [-2.2, -1.6],
    [-1.2, 1.4],
    [0.0, 0.2],
    [1.6, 1.0],
    [2.4, -0.6],
    [0.8, -1.8],
  ];
  for (let k = 0; k < 6; k++) {
    for (let i = 0; i < 80; i++) {
      const cx = centers[k][0];
      const cy = centers[k][1];
      const x = cx + (Math.random() - 0.5) * 1.1;
      const y = cy + (Math.random() - 0.5) * 1.1;
      pts.push({ x, y, k });
    }
  }
  return pts;
}

function makeNetworthSeries(months = 12) {
  const out = [];
  let net = 14500;
  for (let i = 1; i <= months; i++) {
    net += 400 + Math.random() * 350;
    out.push({ month: i, networth: Math.round(net) });
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
      }}
    >
      {/* Blue gradient triangle */}
      <Box
        sx={{
          position: "absolute",
          top: -80,
          right: -80,
          width: 220,
          height: 220,
          background: "linear-gradient(135deg, rgba(59,130,246,0.90), rgba(147,197,253,0.15))",
          clipPath: "polygon(0 0, 100% 0, 100% 100%)",
          filter: "blur(0px)",
          opacity: 0.95,
        }}
      />
      <Box sx={{ position: "relative", height: "100%", p: 2.2 }}>
        <Stack direction="row" alignItems="flex-start" justifyContent="space-between" spacing={2}>
          <Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 800, letterSpacing: 0.2 }}>
              {title}
            </Typography>
            {subtitle ? (
              <Typography variant="caption" sx={{ opacity: 0.75 }}>
                {subtitle}
              </Typography>
            ) : null}
          </Box>
          {actions ? <Box>{actions}</Box> : null}
        </Stack>
        <Divider sx={{ mt: 1.2, mb: 1.6, opacity: 0.25 }} />
        <Box sx={{ height: `calc(100% - 66px)` }}>{children}</Box>
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
    top: 3,
    probs: [0.06, 0.10, 0.12, 0.52, 0.14, 0.06],
  }));

  const radar1 = useMemo(() => makeRadarData(1), [savedAt]);
  const radar2 = useMemo(() => makeRadarData(2), [savedAt]);
  const radar3 = useMemo(() => makeRadarData(3), [savedAt]);
  const radar4 = useMemo(() => makeRadarData(4), [savedAt]);

  const scatter = useMemo(() => makeClusterScatter(), []);
  const networth = useMemo(() => makeNetworthSeries(18), [savedAt]);

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

  function onSave() {
    // UI-only: mimic inference/heuristic update
    const jitter = () => Math.random() * 0.08 - 0.04;
    const base = [0.05, 0.07, 0.10, 0.55, 0.16, 0.07].map((p) => Math.max(0.01, Math.min(0.90, p + jitter())));
    const sum = base.reduce((a, b) => a + b, 0);
    const probs = base.map((p) => p / sum);
    const top = probs.indexOf(Math.max(...probs));

    setClusterResult({ top, probs });
    setSavedAt(new Date());
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

      <Container maxWidth="xl" sx={{ py: 3.2 }}>
        {/* Row 1: 4 radars */}
        <Grid container spacing={2.2}>
          <Grid item xs={12} md={3}>
            <GlassCard title="Radar · Snapshot" subtitle="Last saved month profile" height={300}>
              <RadarPanel data={radar1} />
            </GlassCard>
          </Grid>
          <Grid item xs={12} md={3}>
            <GlassCard title="Radar · Trend" subtitle="Rolling behavior signals" height={300}>
              <RadarPanel data={radar2} />
            </GlassCard>
          </Grid>
          <Grid item xs={12} md={3}>
            <GlassCard title="Radar · Risk" subtitle="Debt & essentials pressure" height={300}>
              <RadarPanel data={radar3} />
            </GlassCard>
          </Grid>
          <Grid item xs={12} md={3}>
            <GlassCard title="Radar · Growth" subtitle="Savings momentum" height={300}>
              <RadarPanel data={radar4} />
            </GlassCard>
          </Grid>

          {/* Row 2: cluster scatter (left) with networth (right) */}
          <Grid item xs={12} md={7}>
            <GlassCard
              title="Cluster space"
              subtitle="All 6 clusters overview (mock)"
              height={340}
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
          <Grid item xs={12} md={5}>
            <GlassCard title="Net worth" subtitle="Savings / net worth projection (mock)" height={340}>
              <NetworthPanel data={networth} />
            </GlassCard>
          </Grid>

          {/* Row 3: input lines */}
          <Grid item xs={12}>
            <GlassCard
              title="Monthly inputs"
              subtitle="Add 1+ months. More months → better trajectory signal."
              height={rows.length <= 2 ? 330 : 420}
              actions={
                <Button
                  variant="contained"
                  onClick={onSave}
                  startIcon={<CheckCircleRoundedIcon />}
                  sx={{
                    borderRadius: 999,
                    textTransform: "none",
                    fontWeight: 900,
                    background: "linear-gradient(135deg, rgba(34,197,94,1), rgba(16,185,129,0.85))",
                    boxShadow: "0 14px 30px rgba(34,197,94,0.22)",
                  }}
                >
                  Save
                </Button>
              }
            >
              <InputsPanel rows={rows} onAdd={addRow} onDelete={deleteRow} onChange={updateCell} />
            </GlassCard>
          </Grid>

          {/* Result footer */}
          <Grid item xs={12}>
            <ResultsPanel clusterResult={clusterResult} />
          </Grid>
        </Grid>
      </Container>

      <Box sx={{ py: 3, opacity: 0.75 }}>
        <Container maxWidth="xl">
          <Typography variant="caption">
            UI prototype · ML wiring comes next (save triggers a mock update).
          </Typography>
        </Container>
      </Box>
    </Box>
  );
}

// -----------------------------
// Charts
// -----------------------------
function RadarPanel({ data }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid strokeOpacity={0.25} />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
        <Radar dataKey="v" stroke="rgba(59,130,246,0.9)" fill="rgba(59,130,246,0.30)" />
        <Tooltip />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function ClusterScatterPanel({ data }) {
  // Split by cluster for legend and styling
  const grouped = useMemo(() => {
    const g = new Map();
    for (const p of data) {
      if (!g.has(p.k)) g.set(p.k, []);
      g.get(p.k).push(p);
    }
    return [...g.entries()].sort((a, b) => a[0] - b[0]);
  }, [data]);

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
      <ScatterChart margin={{ top: 6, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeOpacity={0.25} />
        <XAxis type="number" dataKey="x" tick={{ fontSize: 11 }} />
        <YAxis type="number" dataKey="y" tick={{ fontSize: 11 }} />
        <Tooltip cursor={{ strokeOpacity: 0.2 }} />
        <Legend />
        {grouped.map(([k, pts]) => (
          <Scatter
            key={k}
            name={CLUSTER_META[k]?.name ?? `C${k}`}
            data={pts}
            fill={palette[k % palette.length]}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function NetworthPanel({ data }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid strokeOpacity={0.25} />
        <XAxis dataKey="month" tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} width={40} />
        <Tooltip />
        <Line type="monotone" dataKey="networth" stroke="rgba(59,130,246,0.9)" strokeWidth={3} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// -----------------------------
// Inputs panel
// -----------------------------
function InputsPanel({ rows, onAdd, onDelete, onChange }) {
  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Box sx={{ flex: 1, overflow: "auto", pr: 0.8 }}>
        <Stack spacing={1.8}>
          {rows.map((r, idx) => (
            <MonthRow
              key={idx}
              index={idx}
              row={r}
              canDelete={rows.length > 1}
              onDelete={() => onDelete(idx)}
              onChange={(key, value) => onChange(idx, key, value)}
            />
          ))}
        </Stack>
      </Box>

      <Stack direction="row" alignItems="center" justifyContent="flex-end" sx={{ pt: 1.6 }}>
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
          }}
        >
          Add month
        </Button>
      </Stack>
    </Box>
  );
}

function MonthRow({ index, row, onChange, onDelete, canDelete }) {
  return (
    <Box
      sx={{
        borderRadius: TOKENS.radius,
        border: "1px solid rgba(15, 23, 42, 0.10)",
        background: "rgba(255,255,255,0.70)",
        backdropFilter: `blur(${TOKENS.blur}px)`,
        p: 1.4,
      }}
    >
      <Stack spacing={1.2}>
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
            <IconButton
              onClick={onDelete}
              disabled={!canDelete}
              sx={{
                width: 40,
                height: 40,
                borderRadius: 12,
                border: "1px solid rgba(244,63,94,0.25)",
                background: canDelete ? "rgba(244,63,94,0.10)" : "rgba(2,6,23,0.03)",
              }}
            >
              <DeleteOutlineRoundedIcon sx={{ color: canDelete ? "rgba(244,63,94,0.9)" : "rgba(2,6,23,0.35)" }} />
            </IconButton>
          </Stack>
        </Stack>

        <Grid container spacing={1.2}>
          {CATEGORIES_17.map((c) => (
            <Grid item xs={6} sm={4} md={2.4} lg={2} key={c.key}>
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
function ResultsPanel({ clusterResult }) {
  const top = clusterResult.top;
  const probs = clusterResult.probs;

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
      <Box sx={{ p: 2.2 }}>
        <Stack spacing={1.2}>
          <Stack direction={{ xs: "column", md: "row" }} alignItems={{ xs: "flex-start", md: "center" }} justifyContent="space-between" spacing={1.2}>
            <Box>
              <Typography sx={{ fontWeight: 900 }}>Conclusion</Typography>
              <Typography variant="caption" sx={{ opacity: 0.75 }}>
                After Save, we show the predicted cluster + probability breakdown (mock UI).
              </Typography>
            </Box>

            <Stack direction="row" spacing={1} alignItems="center">
              <Chip
                label={`${CLUSTER_META[top].name} · ${CLUSTER_META[top].range}`}
                sx={{
                  borderRadius: 999,
                  fontWeight: 900,
                  background: "rgba(34,197,94,0.10)",
                  border: "1px solid rgba(34,197,94,0.25)",
                }}
              />
              <Chip
                label={`Top prob: ${(Math.max(...probs) * 100).toFixed(1)}%`}
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

          <Grid container spacing={1.2}>
            {probs.map((p, i) => (
              <Grid item xs={12} sm={6} md={4} lg={2} key={i}>
                <Box
                  sx={{
                    borderRadius: 16,
                    border: "1px solid rgba(15, 23, 42, 0.10)",
                    background: "rgba(255,255,255,0.70)",
                    p: 1.4,
                    position: "relative",
                    overflow: "hidden",
                  }}
                >
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      background: i === top
                        ? "linear-gradient(135deg, rgba(34,197,94,0.14), rgba(34,197,94,0.02))"
                        : "linear-gradient(135deg, rgba(59,130,246,0.10), rgba(59,130,246,0.01))",
                      pointerEvents: "none",
                    }}
                  />
                  <Stack spacing={0.6} sx={{ position: "relative" }}>
                    <Typography variant="caption" sx={{ opacity: 0.75, fontWeight: 800 }}>
                      {CLUSTER_META[i].name}
                    </Typography>
                    <Typography sx={{ fontWeight: 1000, fontSize: 22 }}>
                      {(p * 100).toFixed(2)}%
                    </Typography>
                    <Typography variant="caption" sx={{ opacity: 0.65 }}>
                      {CLUSTER_META[i].range}
                    </Typography>
                  </Stack>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Stack>
      </Box>
    </Card>
  );
}
