const fs = require("node:fs/promises");
const http = require("node:http");
const https = require("node:https");
const path = require("node:path");
const { URL } = require("node:url");

const root = __dirname;
const port = Number.parseInt(process.env.WEB_PORT || "5173", 10);
const apiTarget = process.env.HILBERT_API_URL || "http://127.0.0.1:8000";

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon"
};

const server = http.createServer(async (request, response) => {
  try {
    const requestUrl = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);
    if (requestUrl.pathname.startsWith("/api/")) {
      proxyApi(request, response);
      return;
    }
    await serveStatic(requestUrl.pathname, response);
  } catch (error) {
    response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    response.end(error instanceof Error ? error.message : "Internal server error");
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Project Hilbert web UI: http://127.0.0.1:${port}`);
  console.log(`API proxy target: ${apiTarget}`);
});

async function serveStatic(pathname, response) {
  const decodedPath = decodeURIComponent(pathname);
  const requestedPath = decodedPath === "/" ? "/index.html" : decodedPath;
  const filePath = path.join(root, requestedPath);
  const relative = path.relative(root, filePath);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    response.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Forbidden");
    return;
  }

  try {
    const content = await fs.readFile(filePath);
    const extension = path.extname(filePath).toLowerCase();
    response.writeHead(200, {
      "Content-Type": mimeTypes[extension] || "application/octet-stream",
      "Cache-Control": "no-store"
    });
    response.end(content);
  } catch (error) {
    if (error && error.code === "ENOENT") {
      response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      response.end("Not found");
      return;
    }
    throw error;
  }
}

function proxyApi(clientRequest, clientResponse) {
  const sourceUrl = new URL(clientRequest.url || "/", "http://localhost");
  const targetUrl = new URL(sourceUrl.pathname + sourceUrl.search, apiTarget);
  const transport = targetUrl.protocol === "https:" ? https : http;
  const headers = { ...clientRequest.headers, host: targetUrl.host };
  delete headers.connection;

  const proxyRequest = transport.request(
    {
      protocol: targetUrl.protocol,
      hostname: targetUrl.hostname,
      port: targetUrl.port,
      method: clientRequest.method,
      path: targetUrl.pathname + targetUrl.search,
      headers
    },
    (proxyResponse) => {
      clientResponse.writeHead(proxyResponse.statusCode || 502, proxyResponse.headers);
      proxyResponse.pipe(clientResponse);
    }
  );

  proxyRequest.on("error", (error) => {
    clientResponse.writeHead(502, { "Content-Type": "application/json; charset=utf-8" });
    clientResponse.end(JSON.stringify({ detail: `API proxy failed: ${error.message}` }));
  });

  clientRequest.pipe(proxyRequest);
}
