import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { validateReading } from "@/lib/utils";

import { SensorForm } from "./SensorForm";

const healthy = { engine_rpm: 900, lub_oil_pressure: 3.4, fuel_pressure: 6.2, coolant_pressure: 2.2, lub_oil_temp: 77, coolant_temp: 78 };
const samples = { healthy, faulty: { ...healthy, lub_oil_pressure: 1.2 } };

describe("validateReading", () => {
  it("flags missing, non-numeric and out-of-range values", () => {
    const errs = validateReading({
      engine_rpm: "",
      lub_oil_pressure: "abc",
      fuel_pressure: "-1",
      coolant_pressure: "2",
      lub_oil_temp: "80",
      coolant_temp: "80",
    });
    expect(errs.engine_rpm).toBe("Required");
    expect(errs.lub_oil_pressure).toBe("Enter a number");
    expect(errs.fuel_pressure).toMatch(/between 0 and 50/);
    expect(errs.coolant_pressure).toBeUndefined();
  });
});

describe("SensorForm", () => {
  it("blocks submit and shows inline errors when empty", () => {
    const onSubmit = vi.fn();
    render(<SensorForm onSubmit={onSubmit} />);
    fireEvent.click(screen.getByRole("button", { name: /predict fault risk/i }));
    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getAllByText("Required")).toHaveLength(6);
  });

  it("loads a sample and submits numbers", () => {
    const onSubmit = vi.fn();
    render(<SensorForm onSubmit={onSubmit} samples={samples} />);
    fireEvent.click(screen.getByRole("button", { name: /healthy sample/i }));
    fireEvent.click(screen.getByRole("button", { name: /predict fault risk/i }));
    expect(onSubmit).toHaveBeenCalledWith(healthy);
  });

  it("shows server-side 422 errors next to the field", () => {
    render(
      <SensorForm onSubmit={vi.fn()} serverErrors={[{ field: "coolant_temp", message: "Input should be less than or equal to 250" }]} />,
    );
    expect(screen.getByText(/less than or equal to 250/)).toBeInTheDocument();
  });
});
