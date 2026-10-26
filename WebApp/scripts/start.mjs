import { spawn, fork } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createServer } from 'node:net';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadDeployment, matchesDeployment, assertReusableService } from '../config/deployment.mjs';

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const noOpen = process.argv.includes('--no-open') || process.env.CAMPUS_NO_OPEN === '1';
let server;
let stopping = false;

function run(file, args) {
  return new Promise((accept, reject) => {
    const child = spawn(process.execPath, [file, ...args], {
      cwd: appRoot, shell: false, windowsHide: true, stdio: 'inherit',
    });
    child.once('error', reject);
    child.once('exit', (code) => code === 0 ? accept() : reject(new Error(`Build failed (exit code ${code}).`)));
  });
}

function portAvailable(port) {
  return new Promise((accept) => {
    const probe = createServer();
    probe.once('error', () => accept(false));
    probe.listen(port, '127.0.0.1', () => probe.close(() => accept(true)));
  });
}

function openBrowser(url) {
  const [exe, args] = process.platform === 'win32'
    ? ['rundll32.exe', ['url.dll,FileProtocolHandler', url]]
    : process.platform === 'darwin' ? ['open', [url]] : ['xdg-open', [url]];
  const browser = spawn(exe, args, { shell: false, windowsHide: true, stdio: 'ignore' });
  browser.on('error', () => console.log(`Open in your browser: ${url}`));
  browser.unref();
}

function stop() {
  if (stopping) return;
  stopping = true;
  if (!server || server.exitCode !== null) {
    if (process.connected) process.disconnect();
    return;
  }
  console.log('\nStopping the local service and model worker…');
  if (server.connected) server.send('shutdown');
  else server.kill('SIGTERM');
  const timer = setTimeout(() => server?.kill('SIGKILL'), 8000);
  timer.unref();
}
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
process.on('message', message => { if (message === 'shutdown') stop(); });

async function main() {
  const deployment = loadDeployment();
  if (Number(process.versions.node.split('.')[0]) !== 24) {
    throw new Error('Use Node.js 24 LTS. This application has been tested with Node 24.');
  }
  if (!existsSync(resolve(appRoot, 'node_modules/typescript/bin/tsc'))) {
    throw new Error('Run npm ci in WebApp before starting the application with npm start.');
  }
  console.log('\nCST-MIL · Checking and building the interface…');
  await run(resolve(appRoot, 'node_modules/typescript/bin/tsc'), ['--noEmit']);
  await run(resolve(appRoot, 'node_modules/vite/bin/vite.js'), ['build']);
  const requested = Number(process.env.PORT || 3210);
  if (!Number.isInteger(requested) || requested < 1024 || requested > 65535) {
    throw new Error('PORT must be an integer from 1024 to 65535.');
  }
  let port = requested;
  while (!(await portAvailable(port))) {
    if (!process.env.PORT && !process.env.CAMPUS_DATA_DIR) {
      try {
        const existingUrl = `http://127.0.0.1:${port}`;
        const response = await fetch(`${existingUrl}/api/health`, { signal: AbortSignal.timeout(1000) });
        const health = response.ok ? await response.json() : null;
        if (health?.app === 'Campus Text Lab') {
          assertReusableService(health, deployment);
          console.log(`\nThe application is already running: ${existingUrl}\nPress Ctrl+C in the original launcher to stop it.`);
          if (!noOpen) openBrowser(existingUrl);
          return;
        }
      } catch (error) { if (error instanceof Error && error.message.startsWith('This port is occupied by an unverified or older service')) throw error; /* Another application may own this port. */ }
    }
    if (process.env.PORT || port >= requested + 10) throw new Error(`Port ${port} is occupied. Stop the existing service or set another PORT.`);
    port += 1;
  }
  if (stopping) return;
  const url = `http://127.0.0.1:${port}`;
  server = fork(resolve(appRoot, 'server/index.ts'), [], {
    cwd: appRoot, execArgv: ['--import', 'tsx'],
    env: { ...process.env, PORT: String(port) },
    windowsHide: true, stdio: ['inherit', 'inherit', 'inherit', 'ipc'],
  });
  server.once('error', (error) => { console.error(error.message); process.exitCode = 1; });
  server.once('exit', (code) => { process.exitCode = stopping ? 0 : (code ?? 1); if (process.connected) process.disconnect(); });
  const startupDeadline = Date.now() + 100_000;
  while (Date.now() < startupDeadline && !stopping) {
    if (server.exitCode !== null) throw new Error('The local service could not start. Check the errors above.');
    try {
      const response = await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(1000) });
      if (response.ok) {
        const health = await response.json();
        if (health.worker?.status === 'error') throw new Error(`Model startup failed: ${health.worker.error || 'Check the deployment manifest'}`);
        if (health.worker?.status !== 'ready') { await new Promise((accept) => setTimeout(accept, 500)); continue; }
        if (health.appProtocolVersion !== 2 || !matchesDeployment(health.worker, deployment)) throw new Error('Service model identity does not match the deployment manifest. Startup stopped.');
        console.log(`\n  Browser URL: ${url}\n  Press Ctrl+C to stop the application.\n`);
        if (!noOpen) openBrowser(url);
        return;
      }
    } catch (error) { if (error instanceof Error && /^(Model startup failed|Service model identity)/.test(error.message)) throw error; /* Wait until the owned child serves its health endpoint. */ }
    await new Promise((accept) => setTimeout(accept, 500));
  }
  if (!stopping) { stop(); throw new Error('Startup timed out. Check the Python environment and model files.'); }
}

main().then(() => {
  if (!server && process.connected) process.disconnect();
}).catch((error) => {
  console.error(`\nStartup failed: ${error.message}`);
  process.exitCode = 1;
  stop();
});
