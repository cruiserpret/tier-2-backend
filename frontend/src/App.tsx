import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { TopNav } from "./components/TopNav";
import { BetaBanner } from "./components/BetaBanner";
import { HomeView } from "./views/HomeView";
import { SimulationView } from "./views/SimulationView";
import { ReportView } from "./views/ReportView";

export default function App() {
  return (
    <BrowserRouter>
      <TopNav />
      <BetaBanner />
      <Routes>
        <Route path="/" element={<Navigate to="/dtc-v3" replace />} />
        <Route path="/dtc-v3" element={<HomeView />} />
        <Route path="/dtc-v3/simulation/:id" element={<SimulationView />} />
        <Route path="/dtc-v3/report/:id" element={<ReportView />} />
        <Route path="*" element={<Navigate to="/dtc-v3" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
