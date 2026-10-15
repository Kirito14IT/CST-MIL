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
    child.once('exit', (code) => code === 0 ? accept() : reject(new Error(`构建失败（退出码 ${code}）。`)));
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
  browser.on('error', () => console.log(`请在浏览器打开 ${url}`));
  browser.unref();
}

function stop() {
  if (stopping) return;
  stopping = true;
  if (!server || server.exitCode !== null) {
    if (process.connected) process.disconnect();
    return;
  }
  console.log('\n正在关闭本地服务与算法进程…');
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
    throw new Error('请使用 Node.js 24 LTS（此项目已在 Node 24 验证）。');
  }
  if (!existsSync(resolve(appRoot, 'node_modules/typescript/bin/tsc'))) {
    throw new Error('首次运行请先在 app 目录执行 npm install，然后执行 npm start。');
  }
  console.log('\n校园文本分析 · 正在检查与构建页面…');
  await run(resolve(appRoot, 'node_modules/typescript/bin/tsc'), ['--noEmit']);
  await run(resolve(appRoot, 'node_modules/vite/bin/vite.js'), ['build']);
  const requested = Number(process.env.PORT || 3210);
  if (!Number.isInteger(requested) || requested < 1024 || requested > 65535) {
    throw new Error('PORT 必须是 1024–65535 之间的整数。');
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
          console.log(`\n系统已经在运行：${existingUrl}\n请在原启动窗口按 Ctrl+C 关闭。`);
          if (!noOpen) openBrowser(existingUrl);
          return;
        }
      } catch (error) { if (error instanceof Error && error.message.startsWith('此端口已有旧模型')) throw error; /* Another application may own this port. */ }
    }
    if (process.env.PORT || port >= requested + 10) throw new Error(`端口 ${port} 已占用，请关闭旧服务或设置其他 PORT。`);
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
    if (server.exitCode !== null) throw new Error('本地服务未能启动，请查看上面的错误信息。');
    try {
      const response = await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(1000) });
      if (response.ok) {
        const health = await response.json();
        if (health.worker?.status === 'error') throw new Error(`模型启动失败：${health.worker.error || '请检查部署清单'}`);
        if (health.worker?.status !== 'ready') { await new Promise((accept) => setTimeout(accept, 500)); continue; }
        if (health.appProtocolVersion !== 2 || !matchesDeployment(health.worker, deployment)) throw new Error('启动服务的模型身份与当前部署清单不一致，停止本次启动。');
        console.log(`\n  浏览器地址：${url}\n  按 Ctrl+C 关闭系统。\n`);
        if (!noOpen) openBrowser(url);
        return;
      }
    } catch (error) { if (error instanceof Error && /^(模型启动失败|启动服务的模型身份)/.test(error.message)) throw error; /* Wait until the owned child serves its health endpoint. */ }
    await new Promise((accept) => setTimeout(accept, 500));
  }
  if (!stopping) { stop(); throw new Error('启动超时，请检查 Python 环境和模型文件。'); }
}

main().then(() => {
  if (!server && process.connected) process.disconnect();
}).catch((error) => {
  console.error(`\n启动失败：${error.message}`);
  process.exitCode = 1;
  stop();
});
