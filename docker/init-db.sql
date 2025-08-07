-- Initialize Quantum Task Planner Database

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create quantum_tasks table
CREATE TABLE IF NOT EXISTS quantum_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
    state VARCHAR(20) NOT NULL DEFAULT 'superposition',
    
    -- Quantum properties
    amplitude DECIMAL(5,4) NOT NULL DEFAULT 0.5 CHECK (amplitude >= 0.0 AND amplitude <= 1.0),
    phase DECIMAL(8,6) NOT NULL DEFAULT 0.0 CHECK (phase >= 0.0 AND phase <= 6.283185),
    coherence_time INTEGER NOT NULL DEFAULT 3600,
    entangled_tasks UUID[] DEFAULT '{}',
    
    -- Traditional properties
    estimated_duration DECIMAL(8,2) DEFAULT 1.0 CHECK (estimated_duration >= 0),
    dependencies UUID[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    due_date TIMESTAMP WITH TIME ZONE,
    progress DECIMAL(5,4) DEFAULT 0.0 CHECK (progress >= 0.0 AND progress <= 1.0)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_state ON quantum_tasks (state);
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_priority ON quantum_tasks (priority);
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_created_at ON quantum_tasks (created_at);
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_due_date ON quantum_tasks (due_date);
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_dependencies ON quantum_tasks USING GIN (dependencies);
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_entangled ON quantum_tasks USING GIN (entangled_tasks);
CREATE INDEX IF NOT EXISTS idx_quantum_tasks_metadata ON quantum_tasks USING GIN (metadata);

-- Create quantum_events table for audit trail
CREATE TABLE IF NOT EXISTS quantum_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES quantum_tasks(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_id UUID,
    
    -- Cryptographic integrity
    event_hash VARCHAR(64) NOT NULL,
    merkle_proof JSONB
);

-- Create indexes for events
CREATE INDEX IF NOT EXISTS idx_quantum_events_task_id ON quantum_events (task_id);
CREATE INDEX IF NOT EXISTS idx_quantum_events_type ON quantum_events (event_type);
CREATE INDEX IF NOT EXISTS idx_quantum_events_timestamp ON quantum_events (timestamp);
CREATE INDEX IF NOT EXISTS idx_quantum_events_session ON quantum_events (session_id);

-- Create quantum_sessions table
CREATE TABLE IF NOT EXISTS quantum_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_name VARCHAR(200) NOT NULL,
    planner_name VARCHAR(100) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    configuration JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active'
);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_id UUID REFERENCES quantum_sessions(id),
    
    -- Task metrics
    total_tasks INTEGER DEFAULT 0,
    running_tasks INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    failed_tasks INTEGER DEFAULT 0,
    
    -- Quantum metrics
    coherent_tasks INTEGER DEFAULT 0,
    entangled_pairs INTEGER DEFAULT 0,
    average_amplitude DECIMAL(5,4) DEFAULT 0.0,
    total_collapses INTEGER DEFAULT 0,
    
    -- Performance metrics
    tasks_per_second DECIMAL(10,6) DEFAULT 0.0,
    average_execution_time DECIMAL(10,6) DEFAULT 0.0,
    success_rate DECIMAL(5,4) DEFAULT 0.0,
    
    -- System metrics
    memory_usage_mb DECIMAL(10,2) DEFAULT 0.0,
    cpu_usage_percent DECIMAL(5,2) DEFAULT 0.0
);

-- Create index for performance metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics (timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_session ON performance_metrics (session_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for quantum_tasks
CREATE TRIGGER update_quantum_tasks_updated_at 
    BEFORE UPDATE ON quantum_tasks 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW executable_tasks AS
SELECT * FROM quantum_tasks 
WHERE state IN ('superposition', 'pending') 
AND (dependencies IS NULL OR array_length(dependencies, 1) IS NULL 
     OR NOT EXISTS (
         SELECT 1 FROM quantum_tasks dep 
         WHERE dep.id = ANY(quantum_tasks.dependencies) 
         AND dep.state != 'completed'
     ));

CREATE OR REPLACE VIEW coherent_tasks AS
SELECT *, 
       (EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) < coherence_time) AS is_coherent
FROM quantum_tasks 
WHERE state != 'completed';

-- Create function for quantum interference calculation
CREATE OR REPLACE FUNCTION calculate_interference(
    amplitude1 DECIMAL(5,4),
    phase1 DECIMAL(8,6),
    amplitude2 DECIMAL(5,4), 
    phase2 DECIMAL(8,6)
) RETURNS DECIMAL(8,6) AS $$
BEGIN
    RETURN amplitude1 * amplitude2 * cos(phase1 - phase2);
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO quantum;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO quantum;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO quantum;

-- Insert sample data (for development)
INSERT INTO quantum_tasks (name, description, priority, amplitude, phase)
VALUES 
    ('Sample Task 1', 'A sample quantum task for testing', 'high', 0.8, 0.0),
    ('Sample Task 2', 'Another sample task', 'medium', 0.6, 1.57),
    ('Sample Task 3', 'Low priority sample task', 'low', 0.4, 3.14)
ON CONFLICT DO NOTHING;