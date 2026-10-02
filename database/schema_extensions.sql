-- E1 extensions to Ayush's schema (schema.sql is unchanged).
-- sales_history is loaded from data/derived/sales_simulation.csv by init-db;
-- inventory_transactions and notifications are written by the E1 simulation
-- and the notification engine. Nothing here duplicates demand/inventory/batches.

CREATE TABLE sales_history (
    date DATE NOT NULL,
    sku_id TEXT NOT NULL,
    warehouse_id TEXT NOT NULL,
    units_sold INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    revenue REAL NOT NULL
);
CREATE INDEX idx_sales_sku_warehouse ON sales_history(sku_id, warehouse_id);
CREATE INDEX idx_sales_date ON sales_history(date);

-- Auditable inventory ledger. quantity is the signed change in on-hand stock.
CREATE TABLE inventory_transactions (
    txn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    txn_date DATE NOT NULL,
    sku_id TEXT NOT NULL,
    warehouse_id TEXT NOT NULL,
    batch_id TEXT,
    txn_type TEXT NOT NULL CHECK (txn_type IN ('SALE', 'RECEIPT', 'ADJUSTMENT', 'TRANSFER_OUT')),
    txn_subtype TEXT,
    quantity INTEGER NOT NULL,
    requested_quantity INTEGER,
    unfulfilled_quantity INTEGER NOT NULL DEFAULT 0,
    on_hand_after INTEGER NOT NULL CHECK (on_hand_after >= 0),
    usable_after INTEGER NOT NULL CHECK (usable_after >= 0),
    reference TEXT,
    reason TEXT
);
CREATE INDEX idx_txn_item ON inventory_transactions(sku_id, warehouse_id, txn_date);
CREATE INDEX idx_txn_run ON inventory_transactions(run_id);

-- Notification / alert log. status is what actually happened:
-- SENT (SMTP accepted), OUTBOX (recorded only, not sent), FAILED (send attempted, error stored).
CREATE TABLE notifications (
    notification_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    alert_date DATE NOT NULL,
    sku_id TEXT NOT NULL,
    warehouse_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    recommended_action TEXT,
    channel TEXT NOT NULL,
    recipient TEXT,
    subject TEXT NOT NULL,
    message TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('SENT', 'OUTBOX', 'FAILED')),
    delivery_ref TEXT,
    error TEXT
);
CREATE INDEX idx_notifications_key ON notifications(sku_id, warehouse_id, alert_type, alert_date);
