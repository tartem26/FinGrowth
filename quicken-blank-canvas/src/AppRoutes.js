import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import ReactRoot from "./index";
import FinGrowthDashboard from "./pages/FinGrowthDashboard";

export default function AppRoutes() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ReactRoot />} />
        <Route path="/fin-growth-dashboard" element={<FinGrowthDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}
