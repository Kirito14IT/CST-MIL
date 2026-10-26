import { defineConfig } from '@playwright/test';
import { resolve } from 'node:path';
import { defaultDeploymentPath } from './config/deployment.mjs';

export default defineConfig({
  testDir: './tests/browser',
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  outputDir: process.env.CAMPUS_E2E_OUTPUT_DIR ?? './test-results/browser-artifacts',
  reporter: [['list']],
  use: {
    baseURL: 'http://127.0.0.1:3291',
    channel: 'msedge',
    viewport: { width: 1366, height: 900 },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'node scripts/start.mjs --no-open',
    url: 'http://127.0.0.1:3291/api/health',
    timeout: 150_000,
    reuseExistingServer: false,
    env: { PORT: '3291', CAMPUS_DATA_DIR: process.env.CAMPUS_E2E_DATA_DIR ?? resolve('test-results/unified-browser/data'), CAMPUS_NO_OPEN: '1', CAMPUS_DEPLOYMENT_FILE: defaultDeploymentPath() },
  },
});
