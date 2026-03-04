import { Navigate, Outlet, Route, Routes, BrowserRouter } from "react-router-dom";
import { getApiBaseUrl, getApiKey } from "../auth/storage";
import { RequireAuth } from "../auth/RequireAuth";
import { LoginPage } from "../pages/LoginPage";
import { RunsPage } from "../pages/RunsPage";
import { RunDetailsPage } from "../pages/RunDetailsPage";
import { ItemPage } from "../pages/ItemPage";
import { AppHeader } from "../components/AppHeader";

function ProtectedLayout() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <AppHeader />
      <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <Outlet />
      </main>
    </div>
  );
}

function DefaultRedirect() {
  return <Navigate to={getApiBaseUrl() && getApiKey() ? "/runs" : "/login"} replace />;
}

export function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<RequireAuth />}>
          <Route element={<ProtectedLayout />}>
            <Route path="/runs" element={<RunsPage />} />
            <Route path="/runs/:run_id" element={<RunDetailsPage />} />
            <Route path="/items/:request_id" element={<ItemPage />} />
          </Route>
        </Route>
        <Route path="*" element={<DefaultRedirect />} />
      </Routes>
    </BrowserRouter>
  );
}
