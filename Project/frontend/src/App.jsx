import { useState, useEffect, createContext, useContext } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Link, useNavigate } from 'react-router-dom'

// ==========================================
// API Service
// ==========================================
const API_BASE = 'http://localhost:8001'

const api = {
    async request(endpoint, options = {}) {
        const token = localStorage.getItem('token')

        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...(token && { 'Authorization': `Bearer ${token}` }),
                ...options.headers
            },
            ...options
        }

        const response = await fetch(`${API_BASE}${endpoint}`, config)

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: 'Request failed' }))
            throw new Error(error.detail || 'Request failed')
        }

        return response.json()
    },

    login: (email, password) => api.request('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email, password })
    }),

    register: (email, password, name) => api.request('/auth/register', {
        method: 'POST',
        body: JSON.stringify({ email, password, name })
    }),

    getMe: () => api.request('/auth/me'),

    getPatients: () => api.request('/patients'),

    createPatient: (data) => api.request('/patients', {
        method: 'POST',
        body: JSON.stringify(data)
    }),

    createPrediction: (patientId) => api.request('/predictions', {
        method: 'POST',
        body: JSON.stringify({ patient_id: patientId })
    }),

    getPredictions: () => api.request('/predictions'),

    getDashboard: () => api.request('/dashboard')
}

// ==========================================
// Auth Context
// ==========================================
const AuthContext = createContext(null)

function AuthProvider({ children }) {
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        const token = localStorage.getItem('token')
        if (token) {
            api.getMe()
                .then(setUser)
                .catch(() => localStorage.removeItem('token'))
                .finally(() => setLoading(false))
        } else {
            setLoading(false)
        }
    }, [])

    const login = async (email, password) => {
        const data = await api.login(email, password)
        localStorage.setItem('token', data.access_token)
        setUser(data.user)
        return data
    }

    const register = async (email, password, name) => {
        const data = await api.register(email, password, name)
        localStorage.setItem('token', data.access_token)
        setUser(data.user)
        return data
    }

    const logout = () => {
        localStorage.removeItem('token')
        setUser(null)
    }

    return (
        <AuthContext.Provider value={{ user, login, register, logout, loading }}>
            {children}
        </AuthContext.Provider>
    )
}

const useAuth = () => useContext(AuthContext)

// ==========================================
// Components
// ==========================================

function Navbar() {
    const { user, logout } = useAuth()

    return (
        <nav style={{
            background: 'var(--glass-bg)',
            backdropFilter: 'blur(10px)',
            borderBottom: '1px solid var(--glass-border)',
            padding: '1rem 0',
            position: 'sticky',
            top: 0,
            zIndex: 100
        }}>
            <div className="container flex flex-between" style={{ alignItems: 'center' }}>
                <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{
                        width: '40px',
                        height: '40px',
                        background: 'var(--accent-gradient)',
                        borderRadius: '10px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '1.2rem'
                    }}>
                        🧬
                    </div>
                    <span style={{ fontSize: '1.2rem', fontWeight: '600', color: 'white' }}>
                        HCT Prediction
                    </span>
                </Link>

                {user && (
                    <div className="flex gap-2" style={{ alignItems: 'center' }}>
                        <Link to="/dashboard" className="btn btn-secondary" style={{ padding: '0.5rem 1rem' }}>
                            Dashboard
                        </Link>
                        <Link to="/predict" className="btn btn-primary" style={{ padding: '0.5rem 1rem' }}>
                            New Prediction
                        </Link>
                        <span style={{ color: 'var(--text-secondary)', margin: '0 0.5rem' }}>
                            {user.name}
                        </span>
                        <button onClick={logout} className="btn btn-secondary" style={{ padding: '0.5rem 1rem' }}>
                            Logout
                        </button>
                    </div>
                )}
            </div>
        </nav>
    )
}

function LoginPage() {
    const { login, register } = useAuth()
    const navigate = useNavigate()
    const [isLogin, setIsLogin] = useState(true)
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [name, setName] = useState('')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        setLoading(true)

        try {
            if (isLogin) {
                await login(email, password)
            } else {
                await register(email, password, name)
            }
            navigate('/dashboard')
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '80vh' }}>
            <div className="card" style={{ width: '100%', maxWidth: '400px' }}>
                <div className="text-center mb-4">
                    <div style={{
                        width: '60px',
                        height: '60px',
                        background: 'var(--accent-gradient)',
                        borderRadius: '16px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '1.5rem',
                        margin: '0 auto 1rem'
                    }}>
                        🧬
                    </div>
                    <h2>{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
                    <p className="text-muted mt-1">
                        {isLogin ? 'Sign in to access predictions' : 'Join the HCT prediction platform'}
                    </p>
                </div>

                {error && (
                    <div style={{
                        background: 'rgba(239, 68, 68, 0.2)',
                        color: 'var(--danger)',
                        padding: '0.75rem 1rem',
                        borderRadius: 'var(--radius-md)',
                        marginBottom: '1rem'
                    }}>
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit}>
                    {!isLogin && (
                        <div className="form-group">
                            <label className="form-label">Name</label>
                            <input
                                type="text"
                                className="form-input"
                                value={name}
                                onChange={(e) => setName(e.target.value)}
                                placeholder="Your name"
                                required={!isLogin}
                            />
                        </div>
                    )}

                    <div className="form-group">
                        <label className="form-label">Email</label>
                        <input
                            type="email"
                            className="form-input"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="you@example.com"
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label className="form-label">Password</label>
                        <input
                            type="password"
                            className="form-input"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="••••••••"
                            required
                        />
                    </div>

                    <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={loading}>
                        {loading ? <span className="spinner"></span> : (isLogin ? 'Sign In' : 'Create Account')}
                    </button>
                </form>

                <p className="text-center mt-3 text-muted">
                    {isLogin ? "Don't have an account? " : "Already have an account? "}
                    <button
                        onClick={() => setIsLogin(!isLogin)}
                        style={{ background: 'none', border: 'none', color: 'var(--accent-primary)', cursor: 'pointer' }}
                    >
                        {isLogin ? 'Sign up' : 'Sign in'}
                    </button>
                </p>
            </div>
        </div>
    )
}

function Dashboard() {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        api.getDashboard()
            .then(setData)
            .catch(console.error)
            .finally(() => setLoading(false))
    }, [])

    if (loading) {
        return (
            <div className="page text-center">
                <span className="spinner" style={{ margin: '3rem auto' }}></span>
            </div>
        )
    }

    return (
        <div className="page container fade-in">
            <div className="page-header">
                <h1 className="page-title">Dashboard</h1>
                <p className="page-subtitle">Overview of your HCT survival predictions</p>
            </div>

            <div className="grid grid-3 mb-4">
                <div className="card stat-card">
                    <div className="stat-value">{data?.total_patients || 0}</div>
                    <div className="stat-label">Total Patients</div>
                </div>
                <div className="card stat-card">
                    <div className="stat-value">{data?.total_predictions || 0}</div>
                    <div className="stat-label">Predictions Made</div>
                </div>
                <div className="card stat-card">
                    <div className="stat-value">{data?.risk_distribution?.High || 0}</div>
                    <div className="stat-label">High Risk Cases</div>
                </div>
            </div>

            <div className="grid grid-2">
                <div className="card">
                    <h3 className="card-title mb-3">Risk Distribution</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {['Low', 'Medium', 'High'].map(risk => {
                            const count = data?.risk_distribution?.[risk] || 0
                            const total = (data?.risk_distribution?.Low || 0) + (data?.risk_distribution?.Medium || 0) + (data?.risk_distribution?.High || 0)
                            const pct = total > 0 ? (count / total * 100) : 0

                            return (
                                <div key={risk}>
                                    <div className="flex flex-between mb-1">
                                        <span className={`risk-badge risk-${risk.toLowerCase()}`}>{risk}</span>
                                        <span>{count} ({pct.toFixed(0)}%)</span>
                                    </div>
                                    <div className="progress-bar">
                                        <div
                                            className="progress-fill"
                                            style={{
                                                width: `${pct}%`,
                                                background: risk === 'Low' ? 'var(--risk-low)' :
                                                    risk === 'Medium' ? 'var(--risk-medium)' : 'var(--risk-high)'
                                            }}
                                        ></div>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>

                <div className="card">
                    <h3 className="card-title mb-3">Recent Predictions</h3>
                    {data?.recent_predictions?.length > 0 ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            {data.recent_predictions.map((pred, i) => (
                                <div key={i} className="flex flex-between" style={{
                                    padding: '0.75rem',
                                    background: 'var(--bg-secondary)',
                                    borderRadius: 'var(--radius-md)'
                                }}>
                                    <span>{pred.patient_id}</span>
                                    <span className={`risk-badge risk-${pred.risk_category.toLowerCase()}`}>
                                        {pred.risk_category}
                                    </span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p className="text-muted">No predictions yet</p>
                    )}
                </div>
            </div>
        </div>
    )
}

function PredictPage() {
    const navigate = useNavigate()
    const [step, setStep] = useState(1)
    const [loading, setLoading] = useState(false)
    const [prediction, setPrediction] = useState(null)
    const [formData, setFormData] = useState({
        name: '',
        age_at_hct: 50,
        donor_age: 35,
        year_hct: 2024,
        race_group: 'White',
        dri_score: 'Intermediate',
        conditioning_intensity: 'MAC',
        graft_type: 'Peripheral blood',
        donor_related: 'Unrelated',
        comorbidity_score: 0,
        karnofsky_score: 90
    })

    const handleChange = (e) => {
        const { name, value, type } = e.target
        setFormData(prev => ({
            ...prev,
            [name]: type === 'number' ? parseFloat(value) : value
        }))
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setLoading(true)

        try {
            // Create patient
            const patient = await api.createPatient(formData)

            // Create prediction
            const pred = await api.createPrediction(patient.id)
            setPrediction(pred)
            setStep(2)
        } catch (err) {
            alert(err.message)
        } finally {
            setLoading(false)
        }
    }

    if (step === 2 && prediction) {
        return (
            <div className="page container fade-in">
                <div className="card" style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'center' }}>
                    <div style={{
                        width: '80px',
                        height: '80px',
                        borderRadius: '50%',
                        background: prediction.risk_category === 'Low' ? 'rgba(16, 185, 129, 0.2)' :
                            prediction.risk_category === 'Medium' ? 'rgba(245, 158, 11, 0.2)' :
                                'rgba(239, 68, 68, 0.2)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        margin: '0 auto 1.5rem',
                        fontSize: '2rem'
                    }}>
                        {prediction.risk_category === 'Low' ? '✓' :
                            prediction.risk_category === 'Medium' ? '⚠' : '⚠'}
                    </div>

                    <h2 className="mb-2">Prediction Complete</h2>
                    <p className="text-muted mb-4">Patient: {formData.name}</p>

                    <div style={{
                        padding: '2rem',
                        background: 'var(--bg-secondary)',
                        borderRadius: 'var(--radius-lg)',
                        marginBottom: '1.5rem'
                    }}>
                        <div className="stat-value" style={{ fontSize: '3rem' }}>
                            {(prediction.event_probability * 100).toFixed(1)}%
                        </div>
                        <div className="stat-label">Event Probability</div>

                        <div className="mt-3">
                            <span className={`risk-badge risk-${prediction.risk_category.toLowerCase()}`} style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>
                                {prediction.risk_category} Risk
                            </span>
                        </div>
                    </div>

                    <div className="flex gap-2" style={{ justifyContent: 'center' }}>
                        <button className="btn btn-secondary" onClick={() => navigate('/dashboard')}>
                            Back to Dashboard
                        </button>
                        <button className="btn btn-primary" onClick={() => { setStep(1); setPrediction(null); }}>
                            New Prediction
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="page container fade-in">
            <div className="page-header">
                <h1 className="page-title">New Prediction</h1>
                <p className="page-subtitle">Enter patient clinical data for survival prediction</p>
            </div>

            <form onSubmit={handleSubmit} className="card" style={{ maxWidth: '800px' }}>
                <div className="grid grid-2">
                    <div className="form-group">
                        <label className="form-label">Patient Name</label>
                        <input type="text" name="name" className="form-input" value={formData.name} onChange={handleChange} required />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Age at HCT</label>
                        <input type="number" name="age_at_hct" className="form-input" value={formData.age_at_hct} onChange={handleChange} required />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Donor Age</label>
                        <input type="number" name="donor_age" className="form-input" value={formData.donor_age} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Year of HCT</label>
                        <input type="number" name="year_hct" className="form-input" value={formData.year_hct} onChange={handleChange} />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Race Group</label>
                        <select name="race_group" className="form-input form-select" value={formData.race_group} onChange={handleChange}>
                            <option>White</option>
                            <option>Black or African-American</option>
                            <option>Asian</option>
                            <option>Hispanic or Latino</option>
                            <option>Other</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Disease Risk Index</label>
                        <select name="dri_score" className="form-input form-select" value={formData.dri_score} onChange={handleChange}>
                            <option>Low</option>
                            <option>Intermediate</option>
                            <option>High</option>
                            <option>Very high</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Conditioning Intensity</label>
                        <select name="conditioning_intensity" className="form-input form-select" value={formData.conditioning_intensity} onChange={handleChange}>
                            <option>MAC</option>
                            <option>RIC</option>
                            <option>NMA</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Graft Type</label>
                        <select name="graft_type" className="form-input form-select" value={formData.graft_type} onChange={handleChange}>
                            <option>Peripheral blood</option>
                            <option>Bone marrow</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Comorbidity Score</label>
                        <input type="number" name="comorbidity_score" className="form-input" value={formData.comorbidity_score} onChange={handleChange} min="0" max="20" />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Karnofsky Score</label>
                        <input type="number" name="karnofsky_score" className="form-input" value={formData.karnofsky_score} onChange={handleChange} min="0" max="100" step="10" />
                    </div>
                </div>

                <div className="mt-3">
                    <button type="submit" className="btn btn-primary" disabled={loading}>
                        {loading ? <span className="spinner"></span> : 'Generate Prediction'}
                    </button>
                </div>
            </form>
        </div>
    )
}

function ProtectedRoute({ children }) {
    const { user, loading } = useAuth()

    if (loading) {
        return (
            <div className="page text-center">
                <span className="spinner" style={{ margin: '3rem auto' }}></span>
            </div>
        )
    }

    if (!user) {
        return <Navigate to="/login" />
    }

    return children
}

// ==========================================
// App
// ==========================================

function App() {
    return (
        <AuthProvider>
            <BrowserRouter>
                <Navbar />
                <main style={{ flex: 1 }}>
                    <Routes>
                        <Route path="/login" element={<LoginPage />} />
                        <Route path="/dashboard" element={
                            <ProtectedRoute>
                                <Dashboard />
                            </ProtectedRoute>
                        } />
                        <Route path="/predict" element={
                            <ProtectedRoute>
                                <PredictPage />
                            </ProtectedRoute>
                        } />
                        <Route path="/" element={<Navigate to="/dashboard" />} />
                    </Routes>
                </main>
            </BrowserRouter>
        </AuthProvider>
    )
}

export default App
