import { useState, useEffect, createContext, useContext } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Link, useNavigate } from 'react-router-dom'

// ==========================================
// API Service
// ==========================================
const API_BASE = '/api'

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

        try {
            const response = await fetch(`${API_BASE}${endpoint}`, config)

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Request failed' }))
                
                // Handle specific error codes
                if (response.status === 503) {
                    throw new Error('Service unavailable. The AI model may need training.')
                }
                if (response.status === 504) {
                    throw new Error('Request timeout. The server is taking too long to respond.')
                }
                if (response.status === 401) {
                    localStorage.removeItem('token')
                    window.location.reload()
                    throw new Error('Session expired. Please login again.')
                }
                
                throw new Error(error.detail || 'Request failed')
            }

            return response.json()
        } catch (err) {
            // Handle network errors (Docker down, etc)
            if (err.name === 'TypeError' && err.message === 'Failed to fetch') {
                throw new Error('Cannot connect to server. Please check that Docker is running.')
            }
            throw err
        }
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

    getDashboard: () => api.request('/dashboard'),

    deletePatient: (patientId) => api.request(`/patients/${patientId}`, {
        method: 'DELETE'
    }),

    deletePrediction: (predictionId) => api.request(`/predictions/${predictionId}`, {
        method: 'DELETE'
    })
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
                        <Link to="/patients" className="btn btn-secondary" style={{ padding: '0.5rem 1rem' }}>
                            Patients
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
                                    <div>
                                        <strong>{pred.patient_name || 'Unknown'}</strong>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                            {new Date(pred.created_at).toLocaleDateString()}
                                        </div>
                                    </div>
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
    const [activeTab, setActiveTab] = useState('demographics')
    const [formData, setFormData] = useState({
        // Basic Info
        name: '',
        // Demographics
        age_at_hct: '',
        year_hct: 2024,
        race_group: '',
        ethnicity: '',
        // Donor Info
        donor_age: '',
        donor_related: '',
        sex_match: '',
        // Disease Info
        prim_disease_hct: '',
        dri_score: '',
        cyto_score: '',
        cyto_score_detail: '',
        mrd_hct: '',
        // Transplant Details
        graft_type: '',
        prod_type: '',
        conditioning_intensity: '',
        tbi_status: '',
        in_vivo_tcd: '',
        gvhd_proph: '',
        rituximab: '',
        melphalan_dose: '',
        // HLA Matching - High Res
        hla_match_a_high: '',
        hla_match_b_high: '',
        hla_match_c_high: '',
        hla_match_drb1_high: '',
        hla_match_dqb1_high: '',
        hla_high_res_8: '',
        hla_high_res_10: '',
        // HLA Matching - Low Res
        hla_match_a_low: '',
        hla_match_b_low: '',
        hla_match_c_low: '',
        hla_match_drb1_low: '',
        hla_match_dqb1_low: '',
        hla_low_res_8: '',
        hla_nmdp_6: '',
        // T-Cell Epitope
        tce_match: '',
        tce_imm_match: '',
        tce_div_match: '',
        // CMV
        cmv_status: '',
        // Performance Scores
        karnofsky_score: '',
        comorbidity_score: '',
        // Comorbidities
        cardiac: '',
        arrhythmia: '',
        diabetes: '',
        hepatic_mild: '',
        hepatic_severe: '',
        obesity: '',
        peptic_ulcer: '',
        prior_tumor: '',
        psych_disturb: '',
        pulm_moderate: '',
        pulm_severe: '',
        renal_issue: '',
        rheum_issue: '',
        vent_hist: ''
    })

    const [validationErrors, setValidationErrors] = useState({})

    // Field validation rules
    // Required fields: name (identification), age_at_hct, karnofsky_score, comorbidity_score (critical for model)
    const validationRules = {
        name: { required: true, minLength: 2, maxLength: 100, noNumbers: true },
        age_at_hct: { required: true, min: 0, max: 120, integer: true },
        donor_age: { min: 0, max: 120, integer: true },
        year_hct: { min: 1980, max: new Date().getFullYear() + 1, integer: true },
        karnofsky_score: { required: true, min: 0, max: 100, step: 10 },
        comorbidity_score: { required: true, min: 0, max: 10, integer: true },
        hla_match_a_high: { min: 0, max: 2 },
        hla_match_b_high: { min: 0, max: 2 },
        hla_match_c_high: { min: 0, max: 2 },
        hla_match_drb1_high: { min: 0, max: 2 },
        hla_match_dqb1_high: { min: 0, max: 2 },
        hla_high_res_8: { min: 0, max: 8, integer: true }
    }

    const validateField = (name, value) => {
        const rules = validationRules[name]
        if (!rules) return null

        if (rules.required && (value === '' || value === null || value === undefined)) {
            return 'This field is required'
        }

        if (value === '' || value === null) return null // Optional empty field is OK

        const numValue = parseFloat(value)
        
        if (rules.min !== undefined && numValue < rules.min) {
            return `Minimum value is ${rules.min}`
        }
        if (rules.max !== undefined && numValue > rules.max) {
            return `Maximum value is ${rules.max}`
        }
        if (rules.integer && !Number.isInteger(numValue)) {
            return 'Must be a whole number'
        }
        if (rules.minLength && String(value).length < rules.minLength) {
            return `Minimum ${rules.minLength} characters`
        }
        if (rules.maxLength && String(value).length > rules.maxLength) {
            return `Maximum ${rules.maxLength} characters`
        }
        if (rules.noNumbers && /\d/.test(value)) {
            return 'Numbers are not allowed in names'
        }

        return null
    }

    const handleChange = (e) => {
        const { name, value, type } = e.target
        
        let processedValue = value
        
        // Block numbers in name field
        if (name === 'name') {
            processedValue = value.replace(/[0-9]/g, '')
        }
        
        if (type === 'number') {
            // Block 'e', 'E', '+', '-' in numeric fields (prevent scientific notation)
            if (/[eE+\-]/.test(value)) {
                return // Don't update, ignore the keystroke
            }
            if (value === '') {
                processedValue = ''
            } else {
                const numValue = parseFloat(value)
                // Prevent negative numbers
                if (numValue < 0) {
                    processedValue = 0
                } else {
                    processedValue = numValue
                }
            }
        }

        // Validate and set error
        const error = validateField(name, processedValue)
        setValidationErrors(prev => ({
            ...prev,
            [name]: error
        }))

        setFormData(prev => ({
            ...prev,
            [name]: processedValue
        }))
    }

    // Block 'e', 'E', '+', '-' keys in numeric inputs
    const handleKeyDown = (e) => {
        if (e.target.type === 'number' && ['e', 'E', '+', '-'].includes(e.key)) {
            e.preventDefault()
        }
    }

    // Handle paste - clean pasted content
    const handlePaste = (e) => {
        const { name } = e.target
        
        // For name field, remove numbers from pasted content
        if (name === 'name') {
            e.preventDefault()
            const pastedText = e.clipboardData.getData('text')
            const cleanedText = pastedText.replace(/[0-9]/g, '')
            
            // Insert cleaned text at cursor position
            const input = e.target
            const start = input.selectionStart
            const end = input.selectionEnd
            const currentValue = formData[name] || ''
            const newValue = currentValue.slice(0, start) + cleanedText + currentValue.slice(end)
            
            setFormData(prev => ({ ...prev, [name]: newValue }))
            
            // Validate
            const error = validateField(name, newValue)
            setValidationErrors(prev => ({ ...prev, [name]: error }))
        }
        
        // For number fields, only allow valid numeric paste
        if (e.target.type === 'number') {
            const pastedText = e.clipboardData.getData('text')
            // If pasted text contains letters (except valid chars), prevent
            if (/[a-zA-Z]/.test(pastedText)) {
                e.preventDefault()
            }
        }
    }
    const validateAllRequired = () => {
        const errors = {}
        let isValid = true

        // Check required fields (critical for prediction)
        if (!formData.name || formData.name.trim() === '') {
            errors.name = 'Patient name is required'
            isValid = false
        }
        if (formData.age_at_hct === '' || formData.age_at_hct === null) {
            errors.age_at_hct = 'Age at HCT is required'
            isValid = false
        }
        if (formData.karnofsky_score === '' || formData.karnofsky_score === null) {
            errors.karnofsky_score = 'Karnofsky score is required for accurate prediction'
            isValid = false
        }
        if (formData.comorbidity_score === '' || formData.comorbidity_score === null) {
            errors.comorbidity_score = 'Comorbidity score is required for accurate prediction'
            isValid = false
        }

        // Check all validation rules
        Object.keys(validationRules).forEach(field => {
            const error = validateField(field, formData[field])
            if (error) {
                errors[field] = error
                if (validationRules[field].required) {
                    isValid = false
                }
            }
        })

        setValidationErrors(errors)
        return isValid
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        
        if (!validateAllRequired()) {
            alert('Please fix the validation errors before submitting.')
            setActiveTab('demographics') // Go back to first tab with errors
            return
        }
        
        setLoading(true)

        try {
            // Filter out empty values
            const cleanData = Object.fromEntries(
                Object.entries(formData).filter(([_, v]) => v !== '' && v !== null)
            )
            // Create patient
            const patient = await api.createPatient(cleanData)

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

    const tabs = [
        { id: 'demographics', label: '👤 Demographics', icon: '👤' },
        { id: 'donor', label: '🩸 Donor', icon: '🩸' },
        { id: 'disease', label: '🔬 Disease', icon: '🔬' },
        { id: 'transplant', label: '💉 Transplant', icon: '💉' },
        { id: 'hla', label: '🧬 HLA Matching', icon: '🧬' },
        { id: 'performance', label: '📊 Performance', icon: '📊' },
        { id: 'comorbidities', label: '🏥 Comorbidities', icon: '🏥' }
    ]

    // Helper to render input with error
    const renderInput = (name, label, type = 'text', props = {}) => {
        const error = validationErrors[name]
        const isRequired = validationRules[name]?.required
        return (
            <div className="form-group">
                <label className="form-label">{label}{isRequired ? ' *' : ''}</label>
                <input 
                    type={type} 
                    name={name} 
                    className={`form-input ${error ? 'input-error' : ''}`}
                    value={formData[name]} 
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    onPaste={handlePaste}
                    required={isRequired}
                    {...props}
                />
                {error && <span className="error-text">{error}</span>}
            </div>
        )
    }

    const renderTabContent = () => {
        switch (activeTab) {
            case 'demographics':
                return (
                    <div className="grid grid-2">
                        {renderInput('name', 'Patient Name', 'text', { placeholder: 'Enter patient name', minLength: 2, maxLength: 100 })}
                        {renderInput('age_at_hct', 'Age at HCT', 'number', { min: 0, max: 120, placeholder: 'Years (0-120)' })}
                        {renderInput('year_hct', 'Year of HCT', 'number', { min: 1980, max: 2030, placeholder: '1980-2030' })}
                        <div className="form-group">
                            <label className="form-label">Year of HCT</label>
                            <input type="number" name="year_hct" className="form-input" value={formData.year_hct} onChange={handleChange} min="1980" max="2030" />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Race Group</label>
                            <select name="race_group" className="form-input form-select" value={formData.race_group} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>White</option>
                                <option>Black or African-American</option>
                                <option>Asian</option>
                                <option>Native Hawaiian or other Pacific Islander</option>
                                <option>American Indian or Alaska Native</option>
                                <option>More than one race</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Ethnicity</label>
                            <select name="ethnicity" className="form-input form-select" value={formData.ethnicity} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>Not Hispanic or Latino</option>
                                <option>Hispanic or Latino</option>
                                <option>Non-resident of the U.S.</option>
                            </select>
                        </div>
                    </div>
                )
            case 'donor':
                return (
                    <div className="grid grid-2">
                        <div className="form-group">
                            <label className="form-label">Donor Age</label>
                            <input type="number" name="donor_age" className="form-input" value={formData.donor_age} onChange={handleChange} min="0" max="120" placeholder="Years" />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Donor Relationship</label>
                            <select name="donor_related" className="form-input form-select" value={formData.donor_related} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>Related</option>
                                <option>Unrelated</option>
                                <option>Multiple donor (non-UCB)</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Sex Match</label>
                            <select name="sex_match" className="form-input form-select" value={formData.sex_match} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>M-M</option>
                                <option>F-F</option>
                                <option>F-M</option>
                                <option>M-F</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">CMV Status</label>
                            <select name="cmv_status" className="form-input form-select" value={formData.cmv_status} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>+/+</option>
                                <option>+/-</option>
                                <option>-/+</option>
                                <option>-/-</option>
                            </select>
                        </div>
                    </div>
                )
            case 'disease':
                return (
                    <div className="grid grid-2">
                        <div className="form-group">
                            <label className="form-label">Primary Disease</label>
                            <select name="prim_disease_hct" className="form-input form-select" value={formData.prim_disease_hct} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>AML</option>
                                <option>ALL</option>
                                <option>MDS</option>
                                <option>MPN</option>
                                <option>CML</option>
                                <option>NHL</option>
                                <option>HD</option>
                                <option>PCD</option>
                                <option>SAA</option>
                                <option>Other acute leukemia</option>
                                <option>Other leukemia</option>
                                <option>Solid tumor</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Disease Risk Index (DRI)</label>
                            <select name="dri_score" className="form-input form-select" value={formData.dri_score} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>Low</option>
                                <option>Intermediate</option>
                                <option>High</option>
                                <option>Very high</option>
                                <option>N/A - non-malignant indication</option>
                                <option>N/A - pediatric</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Cytogenetic Score</label>
                            <select name="cyto_score" className="form-input form-select" value={formData.cyto_score} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>Favorable</option>
                                <option>Intermediate</option>
                                <option>Poor</option>
                                <option>Normal</option>
                                <option>Other</option>
                                <option>TBD</option>
                                <option>Not tested</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">MRD at HCT</label>
                            <select name="mrd_hct" className="form-input form-select" value={formData.mrd_hct} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>Negative</option>
                                <option>Positive</option>
                            </select>
                        </div>
                    </div>
                )
            case 'transplant':
                return (
                    <div className="grid grid-2">
                        <div className="form-group">
                            <label className="form-label">Graft Type</label>
                            <select name="graft_type" className="form-input form-select" value={formData.graft_type} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>Peripheral blood</option>
                                <option>Bone marrow</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Product Type</label>
                            <select name="prod_type" className="form-input form-select" value={formData.prod_type} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>PB</option>
                                <option>BM</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Conditioning Intensity</label>
                            <select name="conditioning_intensity" className="form-input form-select" value={formData.conditioning_intensity} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>MAC</option>
                                <option>RIC</option>
                                <option>NMA</option>
                                <option>TBD</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">TBI Status</label>
                            <select name="tbi_status" className="form-input form-select" value={formData.tbi_status} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>No TBI</option>
                                <option>TBI + Cy +- Other</option>
                                <option>TBI +- Other, &lt;=cGy</option>
                                <option>TBI +- Other, &gt;cGy</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">In-vivo T-cell Depletion</label>
                            <select name="in_vivo_tcd" className="form-input form-select" value={formData.in_vivo_tcd} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>Yes</option>
                                <option>No</option>
                            </select>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Rituximab in Conditioning</label>
                            <select name="rituximab" className="form-input form-select" value={formData.rituximab} onChange={handleChange}>
                                <option value="">Select...</option>
                                <option>Yes</option>
                                <option>No</option>
                            </select>
                        </div>
                    </div>
                )
            case 'hla':
                return (
                    <div>
                        <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>High Resolution Matching</h4>
                        <div className="grid grid-3">
                            <div className="form-group">
                                <label className="form-label">HLA-A</label>
                                <input type="number" name="hla_match_a_high" className="form-input" value={formData.hla_match_a_high} onChange={handleChange} step="0.5" min="0" max="2" placeholder="0-2" />
                            </div>
                            <div className="form-group">
                                <label className="form-label">HLA-B</label>
                                <input type="number" name="hla_match_b_high" className="form-input" value={formData.hla_match_b_high} onChange={handleChange} step="0.5" min="0" max="2" placeholder="0-2" />
                            </div>
                            <div className="form-group">
                                <label className="form-label">HLA-C</label>
                                <input type="number" name="hla_match_c_high" className="form-input" value={formData.hla_match_c_high} onChange={handleChange} step="0.5" min="0" max="2" placeholder="0-2" />
                            </div>
                            <div className="form-group">
                                <label className="form-label">HLA-DRB1</label>
                                <input type="number" name="hla_match_drb1_high" className="form-input" value={formData.hla_match_drb1_high} onChange={handleChange} step="0.5" min="0" max="2" placeholder="0-2" />
                            </div>
                            <div className="form-group">
                                <label className="form-label">HLA-DQB1</label>
                                <input type="number" name="hla_match_dqb1_high" className="form-input" value={formData.hla_match_dqb1_high} onChange={handleChange} step="0.5" min="0" max="2" placeholder="0-2" />
                            </div>
                            <div className="form-group">
                                <label className="form-label">8-Locus Match</label>
                                <input type="number" name="hla_high_res_8" className="form-input" value={formData.hla_high_res_8} onChange={handleChange} step="1" min="0" max="8" placeholder="0-8" />
                            </div>
                        </div>
                        <h4 style={{ margin: '1.5rem 0 1rem', color: 'var(--text-secondary)' }}>T-Cell Epitope Matching</h4>
                        <div className="grid grid-3">
                            <div className="form-group">
                                <label className="form-label">TCE Match</label>
                                <select name="tce_match" className="form-input form-select" value={formData.tce_match} onChange={handleChange}>
                                    <option value="">Select...</option>
                                    <option>Fully matched</option>
                                    <option>Permissive</option>
                                    <option>GvH non-permissive</option>
                                    <option>HvG non-permissive</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label className="form-label">TCE Immunogenicity</label>
                                <select name="tce_imm_match" className="form-input form-select" value={formData.tce_imm_match} onChange={handleChange}>
                                    <option value="">Select...</option>
                                    <option>P/P</option>
                                    <option>G/G</option>
                                    <option>H/H</option>
                                    <option>G/B</option>
                                    <option>H/B</option>
                                    <option>P/H</option>
                                    <option>P/G</option>
                                    <option>P/B</option>
                                </select>
                            </div>
                        </div>
                    </div>
                )
            case 'performance':
                return (
                    <div className="grid grid-2">
                        <div className="form-group">
                            <label className="form-label">Karnofsky Performance Score</label>
                            <input type="number" name="karnofsky_score" className="form-input" value={formData.karnofsky_score} onChange={handleChange} min="0" max="100" step="10" placeholder="0-100" />
                            <small style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>100 = Normal, no complaints</small>
                        </div>
                        <div className="form-group">
                            <label className="form-label">Sorror Comorbidity Score</label>
                            <input type="number" name="comorbidity_score" className="form-input" value={formData.comorbidity_score} onChange={handleChange} min="0" max="10" placeholder="0-10" />
                            <small style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>0 = No comorbidities</small>
                        </div>
                    </div>
                )
            case 'comorbidities':
                return (
                    <div className="grid grid-3">
                        {[
                            { name: 'cardiac', label: 'Cardiac' },
                            { name: 'arrhythmia', label: 'Arrhythmia' },
                            { name: 'diabetes', label: 'Diabetes' },
                            { name: 'hepatic_mild', label: 'Hepatic (Mild)' },
                            { name: 'hepatic_severe', label: 'Hepatic (Severe)' },
                            { name: 'obesity', label: 'Obesity' },
                            { name: 'peptic_ulcer', label: 'Peptic Ulcer' },
                            { name: 'prior_tumor', label: 'Prior Solid Tumor' },
                            { name: 'psych_disturb', label: 'Psychiatric' },
                            { name: 'pulm_moderate', label: 'Pulmonary (Moderate)' },
                            { name: 'pulm_severe', label: 'Pulmonary (Severe)' },
                            { name: 'renal_issue', label: 'Renal' },
                            { name: 'rheum_issue', label: 'Rheumatologic' },
                            { name: 'vent_hist', label: 'Ventilation History' }
                        ].map(item => (
                            <div className="form-group" key={item.name}>
                                <label className="form-label">{item.label}</label>
                                <select name={item.name} className="form-input form-select" value={formData[item.name]} onChange={handleChange}>
                                    <option value="">Select...</option>
                                    <option>Yes</option>
                                    <option>No</option>
                                    <option>Not done</option>
                                </select>
                            </div>
                        ))}
                    </div>
                )
            default:
                return null
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
                <p className="page-subtitle">Enter comprehensive patient clinical data for survival prediction</p>
            </div>

            <form onSubmit={handleSubmit} className="card" style={{ maxWidth: '900px' }}>
                {/* Tab Navigation */}
                <div style={{ 
                    display: 'flex', 
                    flexWrap: 'wrap',
                    gap: '0.5rem', 
                    marginBottom: '1.5rem',
                    borderBottom: '1px solid var(--glass-border)',
                    paddingBottom: '1rem'
                }}>
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            type="button"
                            onClick={() => setActiveTab(tab.id)}
                            style={{
                                padding: '0.5rem 1rem',
                                background: activeTab === tab.id ? 'var(--accent-gradient)' : 'var(--bg-secondary)',
                                border: 'none',
                                borderRadius: 'var(--radius-md)',
                                color: activeTab === tab.id ? 'white' : 'var(--text-secondary)',
                                cursor: 'pointer',
                                fontSize: '0.875rem',
                                fontWeight: activeTab === tab.id ? '600' : '400',
                                transition: 'all 0.2s ease'
                            }}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Tab Content */}
                <div style={{ minHeight: '300px' }}>
                    {renderTabContent()}
                </div>

                {/* Navigation and Submit */}
                <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    marginTop: '2rem',
                    paddingTop: '1.5rem',
                    borderTop: '1px solid var(--glass-border)'
                }}>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        {tabs.map((tab, idx) => (
                            <div
                                key={tab.id}
                                style={{
                                    width: '10px',
                                    height: '10px',
                                    borderRadius: '50%',
                                    background: activeTab === tab.id ? 'var(--accent-primary)' : 'var(--bg-tertiary)',
                                    cursor: 'pointer'
                                }}
                                onClick={() => setActiveTab(tab.id)}
                            />
                        ))}
                    </div>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        {activeTab !== 'demographics' && (
                            <button 
                                type="button" 
                                className="btn btn-secondary"
                                onClick={(e) => {
                                    e.preventDefault()
                                    const currentIdx = tabs.findIndex(t => t.id === activeTab)
                                    if (currentIdx > 0) setActiveTab(tabs[currentIdx - 1].id)
                                }}
                            >
                                ← Previous
                            </button>
                        )}
                        {activeTab === 'comorbidities' ? (
                            <button 
                                type="submit" 
                                className="btn btn-primary" 
                                disabled={loading}
                                style={{ minWidth: '180px' }}
                            >
                                {loading ? (
                                    <><span className="spinner"></span> Processing...</>
                                ) : (
                                    '🧬 Generate Prediction'
                                )}
                            </button>
                        ) : (
                            <button 
                                type="button" 
                                className="btn btn-primary"
                                onClick={(e) => {
                                    e.preventDefault()
                                    e.stopPropagation()
                                    const currentIdx = tabs.findIndex(t => t.id === activeTab)
                                    const nextTab = tabs[currentIdx + 1]
                                    if (nextTab) {
                                        setActiveTab(nextTab.id)
                                    }
                                }}
                            >
                                Next →
                            </button>
                        )}
                    </div>
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
// Patients Management Page
// ==========================================

function PatientsPage() {
    const [patients, setPatients] = useState([])
    const [predictions, setPredictions] = useState([])
    const [loading, setLoading] = useState(true)
    const [deleting, setDeleting] = useState(null)
    const [confirmDelete, setConfirmDelete] = useState(null)
    const [activeTab, setActiveTab] = useState('patients')
    const [searchTerm, setSearchTerm] = useState('')

    useEffect(() => {
        loadData()
    }, [])

    const loadData = async () => {
        setLoading(true)
        try {
            const [patientsData, predictionsData] = await Promise.all([
                api.getPatients(),
                api.getPredictions()
            ])
            setPatients(patientsData || [])
            setPredictions(predictionsData || [])
        } catch (err) {
            console.error('Error loading data:', err)
        } finally {
            setLoading(false)
        }
    }

    const handleDeletePatient = async (patientId, patientName) => {
        if (confirmDelete !== patientId) {
            setConfirmDelete(patientId)
            return
        }

        setDeleting(patientId)
        try {
            await api.deletePatient(patientId)
            setPatients(prev => prev.filter(p => p.id !== patientId))
            setPredictions(prev => prev.filter(p => p.patient_id !== patientId))
            setConfirmDelete(null)
        } catch (err) {
            alert(`Error deleting patient: ${err.message}`)
        } finally {
            setDeleting(null)
        }
    }

    const handleDeletePrediction = async (predictionId) => {
        if (confirmDelete !== predictionId) {
            setConfirmDelete(predictionId)
            return
        }

        setDeleting(predictionId)
        try {
            await api.deletePrediction(predictionId)
            setPredictions(prev => prev.filter(p => p.id !== predictionId))
            setConfirmDelete(null)
        } catch (err) {
            alert(`Error deleting prediction: ${err.message}`)
        } finally {
            setDeleting(null)
        }
    }

    const filteredPatients = patients.filter(p => 
        p.name?.toLowerCase().includes(searchTerm.toLowerCase())
    )

    const filteredPredictions = predictions.filter(p =>
        p.patient_name?.toLowerCase().includes(searchTerm.toLowerCase())
    )

    if (loading) {
        return (
            <div className="page text-center">
                <span className="spinner" style={{ margin: '3rem auto' }}></span>
            </div>
        )
    }

    return (
        <div className="page">
            <div className="container">
                <h1 className="page-title">Manage Patients & Predictions</h1>
                <p className="page-subtitle">View, search, and delete patient records</p>

                {/* Search Bar */}
                <div style={{ marginBottom: '1.5rem' }}>
                    <input
                        type="text"
                        placeholder="🔍 Search by patient name..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="form-control"
                        style={{ maxWidth: '400px' }}
                    />
                </div>

                {/* Tabs */}
                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
                    <button
                        onClick={() => setActiveTab('patients')}
                        className={`btn ${activeTab === 'patients' ? 'btn-primary' : 'btn-secondary'}`}
                    >
                        👥 Patients ({patients.length})
                    </button>
                    <button
                        onClick={() => setActiveTab('predictions')}
                        className={`btn ${activeTab === 'predictions' ? 'btn-primary' : 'btn-secondary'}`}
                    >
                        📊 Predictions ({predictions.length})
                    </button>
                </div>

                {/* Patients Tab */}
                {activeTab === 'patients' && (
                    <div className="card">
                        <h3 className="card-title mb-3">Patient Records</h3>
                        {filteredPatients.length > 0 ? (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                {filteredPatients.map(patient => (
                                    <div key={patient.id} style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        padding: '1rem',
                                        background: 'var(--bg-secondary)',
                                        borderRadius: 'var(--radius-md)',
                                        border: confirmDelete === patient.id ? '2px solid var(--danger)' : 'none'
                                    }}>
                                        <div>
                                            <strong>{patient.name}</strong>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                                Age: {patient.age_at_hct} | Karnofsky: {patient.karnofsky_score} | 
                                                Created: {new Date(patient.created_at).toLocaleDateString()}
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => handleDeletePatient(patient.id, patient.name)}
                                            disabled={deleting === patient.id}
                                            className="btn"
                                            style={{
                                                background: confirmDelete === patient.id ? 'var(--danger)' : 'var(--bg-tertiary)',
                                                color: confirmDelete === patient.id ? 'white' : 'var(--text-primary)',
                                                padding: '0.5rem 1rem',
                                                minWidth: '100px'
                                            }}
                                        >
                                            {deleting === patient.id ? (
                                                <span className="spinner" style={{ width: '16px', height: '16px' }}></span>
                                            ) : confirmDelete === patient.id ? (
                                                '⚠️ Confirm?'
                                            ) : (
                                                '🗑️ Delete'
                                            )}
                                        </button>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-muted">
                                {searchTerm ? 'No patients match your search' : 'No patients registered yet'}
                            </p>
                        )}
                    </div>
                )}

                {/* Predictions Tab */}
                {activeTab === 'predictions' && (
                    <div className="card">
                        <h3 className="card-title mb-3">Prediction History</h3>
                        {filteredPredictions.length > 0 ? (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                {filteredPredictions.map(pred => (
                                    <div key={pred.id} style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        padding: '1rem',
                                        background: 'var(--bg-secondary)',
                                        borderRadius: 'var(--radius-md)',
                                        border: confirmDelete === pred.id ? '2px solid var(--danger)' : 'none'
                                    }}>
                                        <div style={{ flex: 1 }}>
                                            <strong>{pred.patient_name || 'Unknown'}</strong>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                                {new Date(pred.created_at).toLocaleString()}
                                            </div>
                                        </div>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                                            <span className={`risk-badge risk-${pred.risk_category?.toLowerCase()}`}>
                                                {pred.risk_category} ({(pred.event_probability * 100).toFixed(0)}%)
                                            </span>
                                            <button
                                                onClick={() => handleDeletePrediction(pred.id)}
                                                disabled={deleting === pred.id}
                                                className="btn"
                                                style={{
                                                    background: confirmDelete === pred.id ? 'var(--danger)' : 'var(--bg-tertiary)',
                                                    color: confirmDelete === pred.id ? 'white' : 'var(--text-primary)',
                                                    padding: '0.5rem 1rem',
                                                    minWidth: '100px'
                                                }}
                                            >
                                                {deleting === pred.id ? (
                                                    <span className="spinner" style={{ width: '16px', height: '16px' }}></span>
                                                ) : confirmDelete === pred.id ? (
                                                    '⚠️ Confirm?'
                                                ) : (
                                                    '🗑️ Delete'
                                                )}
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-muted">
                                {searchTerm ? 'No predictions match your search' : 'No predictions made yet'}
                            </p>
                        )}
                    </div>
                )}

                {/* Cancel confirmation on click outside */}
                {confirmDelete && (
                    <div 
                        style={{ 
                            position: 'fixed', 
                            bottom: '2rem', 
                            left: '50%', 
                            transform: 'translateX(-50%)',
                            background: 'var(--bg-primary)',
                            padding: '1rem 2rem',
                            borderRadius: 'var(--radius-lg)',
                            boxShadow: 'var(--shadow-lg)',
                            border: '1px solid var(--glass-border)'
                        }}
                    >
                        <span style={{ marginRight: '1rem' }}>Click the button again to confirm deletion</span>
                        <button 
                            onClick={() => setConfirmDelete(null)}
                            className="btn btn-secondary"
                        >
                            Cancel
                        </button>
                    </div>
                )}
            </div>
        </div>
    )
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
                        <Route path="/patients" element={
                            <ProtectedRoute>
                                <PatientsPage />
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
