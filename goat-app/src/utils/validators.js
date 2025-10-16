
export const isPositiveNumber = (v) => { if (v === undefined || v === null || v === "") return false; const n = Number(v); return Number.isFinite(n) && n > 0; };
export const required = (v) => v !== undefined && v !== null && String(v).trim().length > 0;
