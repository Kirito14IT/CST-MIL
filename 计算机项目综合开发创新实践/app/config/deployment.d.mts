import type { DeploymentManifest, ModelIdentity } from '../shared/types.ts';
export interface LoadedDeployment extends DeploymentManifest { deploymentHash: string; manifestPath: string; modelAbsolutePath: string; encoderAbsolutePath: string | null }
export const APP_ROOT: string;
export const REPO_ROOT: string;
export function resolveRepoPath(relative: string): string;
export function defaultDeploymentPath(): string;
export function loadDeployment(file?: string): LoadedDeployment;
export function deploymentIdentity(deployment: LoadedDeployment): ModelIdentity;
export function matchesDeployment(value: unknown, deployment: LoadedDeployment): boolean;
export function assertReusableService(health: unknown, deployment: LoadedDeployment): void;
