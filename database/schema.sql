-- MedCare Pharma schema (Ayush). SQLite dialect.
-- Tables are recreated from data/*.csv by `python main.py init-db`.

CREATE TABLE demand (date DATE NOT NULL, sku_id TEXT NOT NULL, warehouse_id TEXT NOT NULL, demand_qty INTEGER NOT NULL, promotion_flag INTEGER DEFAULT 0, season_flag INTEGER DEFAULT 0);

CREATE TABLE inventory (sku_id TEXT NOT NULL, warehouse_id TEXT NOT NULL, current_stock INTEGER NOT NULL, min_threshold INTEGER NOT NULL, safety_stock INTEGER NOT NULL, capacity INTEGER NOT NULL, lead_time_days INTEGER NOT NULL);

CREATE TABLE batches (batch_id TEXT PRIMARY KEY, sku_id TEXT NOT NULL, warehouse_id TEXT NOT NULL, quantity INTEGER NOT NULL, expiry_date DATE NOT NULL);

CREATE INDEX idx_demand_sku_warehouse ON demand(sku_id, warehouse_id);
CREATE INDEX idx_demand_date ON demand(date);
CREATE INDEX idx_inventory_sku_warehouse ON inventory(sku_id, warehouse_id);
CREATE INDEX idx_batches_sku_warehouse ON batches(sku_id, warehouse_id);
CREATE INDEX idx_batches_expiry ON batches(expiry_date);
