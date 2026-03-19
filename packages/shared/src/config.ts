import { z } from "zod";

type RefinementCtx = z.RefinementCtx;

function requireField(ctx: RefinementCtx, value: unknown, path: string, message: string): void {
	if (!value) {
		ctx.addIssue({ code: z.ZodIssueCode.custom, message, path: [path] });
	}
}

const envSchema = z
	.object({
		// Deployment
		DEPLOYMENT_MODE: z.enum(["selfhosted", "managed"]).default("selfhosted"),

		// Telegram
		TELEGRAM_BOT_TOKEN: z.string().min(1).optional(),
		TELEGRAM_BOT_USERNAME: z.string().optional(),

		// Slack (legacy / managed)
		SLACK_BOT_TOKEN: z.string().startsWith("xoxb-").optional(),
		SLACK_APP_TOKEN: z.string().startsWith("xapp-").optional(),
		SLACK_SIGNING_SECRET: z.string().min(1).optional(),

		// Slack OAuth (managed mode)
		SLACK_CLIENT_ID: z.string().optional(),
		SLACK_CLIENT_SECRET: z.string().optional(),
		SLACK_STATE_SECRET: z.string().optional(),
		BASE_URL: z.string().url().optional(),

		// Dashboard auth
		DASHBOARD_AUTH_MODE: z.enum(["basic", "slack-oauth"]).optional(),
		DASHBOARD_USERNAME: z.string().default("admin"),
		DASHBOARD_PASSWORD: z.string().optional(),

		// Encryption
		ENCRYPTION_KEY: z
			.string()
			.regex(/^[0-9a-fA-F]{64}$/, "ENCRYPTION_KEY must be a 64-character hex string")
			.optional(),

		// LLM
		ANTHROPIC_API_KEY: z.string().min(1).optional(),
		OPENAI_API_KEY: z.string().optional(),
		GOOGLE_AI_API_KEY: z.string().optional(),
		OLLAMA_BASE_URL: z.string().url().optional(),
		DEFAULT_MODEL: z.string().default("claude-sonnet-4-20250514"),
		MAX_TOKENS: z.coerce.number().default(4096),

		// Database
		DATABASE_URL: z.string().url(),

		// Redis (optional for single-instance)
		REDIS_URL: z.string().url().optional(),

		// Application
		LOG_LEVEL: z.enum(["debug", "info", "warn", "error"]).default("info"),
		NODE_ENV: z.enum(["development", "production", "test"]).default("development"),

		// Limits
		MAX_CONCURRENT_RUNS: z.coerce.number().default(16),
		TOOL_TIMEOUT_MS: z.coerce.number().default(600_000),
		AGENT_TIMEOUT_MS: z.coerce.number().default(300_000),
		BASH_DEFAULT_TIMEOUT_MS: z.coerce.number().default(120_000),
		STALE_THREAD_TIMEOUT_MS: z.coerce.number().int().min(60_000).default(86_400_000),
		STALE_CHECK_INTERVAL_MS: z.coerce.number().int().min(10_000).default(900_000),
		THREAD_LOCK_TIMEOUT_MS: z.coerce.number().int().min(10_000).default(300_000),
		GITHUB_TOKEN: z.string().optional(),
		BROWSERBASE_API_KEY: z.string().optional(),
		BROWSERBASE_PROJECT_ID: z.string().optional(),
		LINKUP_API_KEY: z.string().optional(),
		CONTEXT7_BASE_URL: z.string().url().optional(),
		SEARCH_API_KEY: z.string().optional(),
		IMAGEN_API_KEY: z.string().optional(),

		// Tool gateway
		TOOL_GATEWAY_PORT: z.coerce.number().default(3001),

		// Tool backend
		TOOL_BACKEND: z.enum(["local", "modal"]).default("local"),
		MODAL_ENDPOINT_URL: z.string().url().optional(),
		MODAL_AUTH_TOKEN: z.string().min(1).optional(),

		// Global usage cap (managed mode) — hard stop across all workspaces
		GLOBAL_MONTHLY_BUDGET_CENTS: z.coerce.number().int().default(5000), // $50 default

		// Cron scheduler
		CRON_CHECK_INTERVAL_MS: z.coerce.number().int().min(1000).default(30_000),
		HEARTBEAT_ENABLED: z
			.enum(["true", "false"])
			.default("true")
			.transform((v) => v === "true"),

		// Spaces
		CONVEX_ACCESS_TOKEN: z.string().optional(),
		CONVEX_TEAM_ID: z.string().optional(),
		VERCEL_TOKEN: z.string().optional(),
		VERCEL_ORG_ID: z.string().optional(),
		SPACES_DOMAIN: z.string().default("apps.openviktor.com"),
		RESEND_API_KEY: z.string().optional(),

		// Pipedream Connect
		PIPEDREAM_CLIENT_ID: z.string().optional(),
		PIPEDREAM_CLIENT_SECRET: z.string().optional(),
		PIPEDREAM_PROJECT_ID: z.string().optional(),
		PIPEDREAM_ENVIRONMENT: z.enum(["development", "production"]).default("development"),

		// Permissions
		DANGEROUSLY_SKIP_PERMISSIONS: z
			.enum(["true", "false"])
			.default("false")
			.transform((v) => v === "true"),

		// Dashboard
		ENABLE_DASHBOARD: z
			.enum(["true", "false"])
			.default("true")
			.transform((v) => v === "true"),
	})
	.superRefine((data, ctx) => {
		const mode = data.DEPLOYMENT_MODE;

		if (mode === "selfhosted") {
			requireField(
				ctx,
				data.TELEGRAM_BOT_TOKEN,
				"TELEGRAM_BOT_TOKEN",
				"TELEGRAM_BOT_TOKEN is required in selfhosted mode",
			);
		}

		{
			const model = data.DEFAULT_MODEL;
			if (model.startsWith("claude-")) {
				requireField(
					ctx,
					data.ANTHROPIC_API_KEY,
					"ANTHROPIC_API_KEY",
					"ANTHROPIC_API_KEY is required when DEFAULT_MODEL is a Claude model",
				);
			} else if (model.startsWith("gpt-")) {
				requireField(
					ctx,
					data.OPENAI_API_KEY,
					"OPENAI_API_KEY",
					"OPENAI_API_KEY is required when DEFAULT_MODEL is an OpenAI model",
				);
			} else if (model.startsWith("gemini-")) {
				requireField(
					ctx,
					data.GOOGLE_AI_API_KEY,
					"GOOGLE_AI_API_KEY",
					"GOOGLE_AI_API_KEY is required when DEFAULT_MODEL is a Gemini model",
				);
			} else if (!model.startsWith("ollama/")) {
				requireField(
					ctx,
					data.ANTHROPIC_API_KEY,
					"ANTHROPIC_API_KEY",
					"ANTHROPIC_API_KEY is required (set DEFAULT_MODEL=ollama/<model> to use a local LLM)",
				);
			}
		}

		if (mode === "managed") {
			requireField(
				ctx,
				data.SLACK_CLIENT_ID,
				"SLACK_CLIENT_ID",
				"SLACK_CLIENT_ID is required in managed mode",
			);
			requireField(
				ctx,
				data.SLACK_CLIENT_SECRET,
				"SLACK_CLIENT_SECRET",
				"SLACK_CLIENT_SECRET is required in managed mode",
			);
			requireField(
				ctx,
				data.SLACK_STATE_SECRET,
				"SLACK_STATE_SECRET",
				"SLACK_STATE_SECRET is required in managed mode",
			);
			requireField(
				ctx,
				data.BASE_URL,
				"BASE_URL",
				"BASE_URL is required in managed mode (public URL for Events API)",
			);
			requireField(
				ctx,
				data.ENCRYPTION_KEY,
				"ENCRYPTION_KEY",
				"ENCRYPTION_KEY is required in managed mode (for encrypting OAuth tokens)",
			);
		}

		const authMode = data.DASHBOARD_AUTH_MODE ?? "basic";
		if (authMode === "basic") {
			requireField(
				ctx,
				data.DASHBOARD_PASSWORD,
				"DASHBOARD_PASSWORD",
				"DASHBOARD_PASSWORD is required when DASHBOARD_AUTH_MODE=basic",
			);
		}

		if (data.TOOL_BACKEND === "modal") {
			requireField(
				ctx,
				data.MODAL_ENDPOINT_URL,
				"MODAL_ENDPOINT_URL",
				"MODAL_ENDPOINT_URL is required when TOOL_BACKEND=modal",
			);
		}

		const pdFields = [
			data.PIPEDREAM_CLIENT_ID,
			data.PIPEDREAM_CLIENT_SECRET,
			data.PIPEDREAM_PROJECT_ID,
		];
		const pdSet = pdFields.filter(Boolean).length;
		if (pdSet > 0 && pdSet < 3) {
			ctx.addIssue({
				code: z.ZodIssueCode.custom,
				message:
					"All three PIPEDREAM_CLIENT_ID, PIPEDREAM_CLIENT_SECRET, and PIPEDREAM_PROJECT_ID must be set together",
				path: ["PIPEDREAM_CLIENT_ID"],
			});
		}
	});

export type EnvConfig = z.infer<typeof envSchema>;

export type DeploymentMode = "selfhosted" | "managed";

let cachedConfig: EnvConfig | null = null;

export function loadConfig(env: Record<string, string | undefined> = process.env): EnvConfig {
	if (cachedConfig) return cachedConfig;
	const result = envSchema.safeParse(env);
	if (!result.success) {
		const formatted = result.error.issues
			.map((issue) => `  ${issue.path.join(".")}: ${issue.message}`)
			.join("\n");
		throw new Error(`Invalid environment configuration:\n${formatted}`);
	}
	cachedConfig = result.data;
	return cachedConfig;
}

export function resetConfig(): void {
	cachedConfig = null;
}

export function isManaged(config?: EnvConfig): boolean {
	const cfg = config ?? cachedConfig;
	if (!cfg) throw new Error("Config not loaded. Call loadConfig() first.");
	return cfg.DEPLOYMENT_MODE === "managed";
}

export function isSelfHosted(config?: EnvConfig): boolean {
	return !isManaged(config);
}

export function getDashboardAuthMode(config?: EnvConfig): "basic" | "slack-oauth" {
	const cfg = config ?? cachedConfig;
	if (!cfg) throw new Error("Config not loaded. Call loadConfig() first.");
	return cfg.DASHBOARD_AUTH_MODE ?? "basic";
}
