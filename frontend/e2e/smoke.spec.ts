import { expect, test, type Route } from "@playwright/test";

const healthy = { engine_rpm: 900, lub_oil_pressure: 3.4, fuel_pressure: 6.2, coolant_pressure: 2.2, lub_oil_temp: 77, coolant_temp: 78 };
const faulty = {
  engine_rpm: 426,
  lub_oil_pressure: 1.88,
  fuel_pressure: 3.72,
  coolant_pressure: 2.25,
  lub_oil_temp: 75.4,
  coolant_temp: 77.4,
};

const record = {
  id: 7,
  created_at: "2026-11-21T10:00:00Z",
  inputs: faulty,
  probability: 0.71,
  label: "faulty",
  risk_band: "high",
  model_version: "pytorch-test",
  top_features: [
    { feature: "lub_oil_pressure", label: "Lubricant oil pressure", value: 1.88, impact: 0.12, deviation: -1.97, direction: "low" },
    { feature: "engine_rpm", label: "Engine RPM", value: 426, impact: 0.08, deviation: -1.29, direction: "low" },
    { feature: "fuel_pressure", label: "Fuel pressure", value: 3.72, impact: 0.02, deviation: -0.8, direction: "low" },
  ],
};

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

test.beforeEach(async ({ page }) => {
  await page.route("**/api/health", (r) =>
    json(r, {
      status: "ok",
      model_loaded: true,
      model_version: "pytorch-test",
      vector_store: { ready: true, chunks: 8 },
      llm_provider: "fake",
      guardrail: "nli",
      database: true,
    }),
  );
  await page.route("**/api/model/info", (r) =>
    json(r, {
      model_version: "pytorch-test",
      served_model: "PyTorch",
      threshold: 0.32,
      risk_bands: { medium: 0.19, high: 0.32 },
      samples: { healthy, faulty },
      sensor_ranges: {},
      sensor_labels: {},
      dataset: { source: "test", rows: 1, fault_share: 0.5 },
    }),
  );
  await page.route("**/api/predict", (r) => json(r, record, 201));
  await page.route("**/api/predictions/7", (r) => json(r, { ...record, messages: [] }));
  await page.route("**/api/predictions?*", (r) => json(r, { items: [record], total: 1, page: 1, page_size: 15 }));
  await page.route("**/api/diagnose", (r) =>
    json(r, {
      answer: "Low oil pressure can mean the oil level is low [Owner's Manual, page 42].",
      citations: [{ source: "Owner's Manual", page: 42 }],
      snippets: [
        { id: "a", source: "Owner's Manual", page: 42, text: "If the oil pressure light comes on, check the oil level.", score: 0.8 },
      ],
      sentences: [{ text: "Low oil pressure can mean the oil level is low [Owner's Manual, page 42].", supported: true, score: 0.9 }],
      supported: true,
      covered: true,
      mode: "groq",
      message_id: 2,
    }),
  );
});

test("load sample → predict → result → cited answer", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /faulty sample/i }).click();
  await page.getByRole("button", { name: /predict fault risk/i }).click();

  const result = page.getByTestId("result");
  await expect(result).toBeVisible();
  await expect(result).toContainText("71%");
  await expect(page.getByText("High risk")).toBeVisible();

  await page.getByRole("button", { name: /send question/i }).click();
  const answer = page.getByTestId("assistant-message");
  await expect(answer).toContainText("Low oil pressure can mean");
  await answer.getByRole("button", { name: /Owner's Manual, page 42/ }).click();
  await expect(page.getByRole("dialog")).toContainText("check the oil level");
});

test("history lists predictions", async ({ page }) => {
  await page.goto("/history");
  await expect(page.getByRole("cell", { name: "Lubricant oil pressure (low)" })).toBeVisible();
});
