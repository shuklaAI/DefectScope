const configuredBase = (import.meta.env.VITE_API_BASE_URL || "").trim();
let BASE = "";
let baseConfigError = null;

if (configuredBase) {
  try {
    const parsed = new URL(configuredBase);
    if (
      !["http:", "https:"].includes(parsed.protocol) ||
      parsed.pathname !== "/" ||
      parsed.search ||
      parsed.hash ||
      parsed.username ||
      parsed.password
    ) {
      throw new Error("Expected an HTTP(S) backend origin without a path.");
    }
    BASE = parsed.origin;
  } catch (error) {
    baseConfigError =
      `Invalid VITE_API_BASE_URL. Set it to the backend origin, such as ` +
      `https://api.example.org. ${error.message}`;
  }
}

function ensureValidBase() {
  if (baseConfigError) throw new Error(baseConfigError);
}

const apiUrl = (path) => {
  ensureValidBase();
  return `${BASE}${path}`;
};

export function assetUrl(path) {
  if (!path) return path;
  ensureValidBase();
  const normalized = path.replace(/^\/results(?=\/)/, "/inspection-assets");
  if (/^https?:\/\//i.test(normalized)) {
    const url = new URL(normalized);
    url.pathname = url.pathname.replace(/^\/results(?=\/)/, "/inspection-assets");
    return url.toString();
  }
  return `${BASE}${normalized.startsWith("/") ? "" : "/"}${normalized}`;
}

// fetch() rejects with an opaque "Failed to fetch" when the API server is not
// running; replace that with an actionable message.
async function request(url, options) {
  try {
    return await fetch(url, options);
  } catch (_) {
    throw new Error(
      "Cannot reach the backend API. Make sure it is running " +
      "(from backend/: uvicorn app.main:app --port 8000). For a separate deployment, set VITE_API_BASE_URL to the backend origin."
    );
  }
}

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch (_) {
      /* ignore parse failure */
    }
    throw new Error(detail || `Request failed with ${res.status}`);
  }
  return res.json();
}

export async function inspectImage(file, mode = "demo") {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  const res = await request(apiUrl("/api/inspect"), { method: "POST", body: form });
  return handle(res);
}

export async function listInspections(limit = 50, mode = null) {
  const q = mode ? `?limit=${limit}&mode=${mode}` : `?limit=${limit}`;
  const res = await request(apiUrl(`/api/inspections${q}`));
  return handle(res);
}

export async function getInspection(id) {
  const res = await request(apiUrl(`/api/inspections/${id}`));
  return handle(res);
}

export async function getDashboard() {
  const res = await request(apiUrl("/api/dashboard"));
  return handle(res);
}

export async function getModelInfo() {
  const res = await request(apiUrl("/api/model/info"));
  return handle(res);
}

export async function checkHealth() {
  const res = await request(apiUrl("/api/health"));
  return handle(res);
}

export async function downloadAsset(path, filename) {
  const res = await request(assetUrl(path));
  if (!res.ok) {
    throw new Error(`Could not download image: ${res.status} ${res.statusText}`);
  }
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}
