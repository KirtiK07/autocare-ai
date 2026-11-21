import { afterEach, describe, expect, it, vi } from "vitest";

import { api, ApiError } from "./api";

const mockFetch = (status: number, body: unknown) =>
  vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValue(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("posts JSON readings to /api/predict", async () => {
    const spy = mockFetch(201, { id: 1, probability: 0.7 });
    const res = await api.predict({
      engine_rpm: 700,
      lub_oil_pressure: 2,
      fuel_pressure: 5,
      coolant_pressure: 2,
      lub_oil_temp: 80,
      coolant_temp: 80,
    });
    expect(res.id).toBe(1);
    const [url, init] = spy.mock.calls[0];
    expect(url).toBe("/api/predict");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body)).engine_rpm).toBe(700);
  });

  it("maps 422 field errors", async () => {
    mockFetch(422, { error: "validation_error", detail: [{ field: "lub_oil_pressure", message: "must be >= 0" }] });
    const err = await api.health().catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).fields[0].field).toBe("lub_oil_pressure");
    expect((err as ApiError).status).toBe(422);
  });

  it("uses the detail string for service errors", async () => {
    mockFetch(503, { error: "llm_timeout", detail: "The assistant took too long." });
    await expect(api.diagnose("why?", 1)).rejects.toMatchObject({
      status: 503,
      code: "llm_timeout",
      message: "The assistant took too long.",
    });
  });

  it("reports network failures", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(api.health()).rejects.toMatchObject({ code: "network_error" });
  });

  it("builds the history query string", async () => {
    const spy = mockFetch(200, { items: [], total: 0, page: 2, page_size: 15 });
    await api.predictions(2, 15, "high");
    expect(spy.mock.calls[0][0]).toBe("/api/predictions?page=2&page_size=15&risk_band=high");
  });
});
