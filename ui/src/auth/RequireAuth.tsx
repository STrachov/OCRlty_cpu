import { Navigate, Outlet, useLocation } from "react-router-dom";
import { getApiBaseUrl, getApiKey } from "./storage";

export function RequireAuth() {
  const location = useLocation();
  const hasAuth = Boolean(getApiBaseUrl() && getApiKey());

  if (!hasAuth) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  return <Outlet />;
}
