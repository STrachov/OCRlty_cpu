import { Navigate, Route, Routes, BrowserRouter } from "react-router-dom";
import { getApiBaseUrl, getApiKey } from "../auth/storage";
import { RequireAuth } from "../auth/RequireAuth";
import { LoginPage } from "../pages/LoginPage";
import { CreateRunPage } from "../pages/CreateRunPage";
import { RunsPage } from "../pages/RunsPage";
import { RunDetailsPage } from "../pages/RunDetailsPage";
import { ItemPage } from "../pages/ItemPage";
import { SettingsPage } from "../pages/SettingsPage";
import { GroundTruthsPage } from "../pages/GroundTruthsPage";
import { CreateGroundTruthPage } from "../pages/CreateGroundTruthPage";
import { GroundTruthEditorPage } from "../pages/GroundTruthEditorPage";
import { AppLayout } from "../layout/AppLayout";

function DefaultRedirect() {
  return <Navigate to={getApiBaseUrl() && getApiKey() ? "/runs" : "/login"} replace />;
}

export function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<RequireAuth />}>
          <Route element={<AppLayout />}>
            <Route path="/runs" element={<RunsPage />} />
            <Route path="/runs/new" element={<CreateRunPage />} />
            <Route path="/runs/:run_id" element={<RunDetailsPage />} />
            <Route path="/ground-truths" element={<GroundTruthsPage />} />
            <Route path="/ground-truths/new" element={<CreateGroundTruthPage />} />
            <Route path="/ground-truths/:gt_id" element={<GroundTruthEditorPage />} />
            <Route path="/items/:request_id" element={<ItemPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Route>
        </Route>
        <Route path="*" element={<DefaultRedirect />} />
      </Routes>
    </BrowserRouter>
  );
}
