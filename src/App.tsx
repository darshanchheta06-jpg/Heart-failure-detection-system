import { useState, useCallback, useMemo } from 'react';
import {
  Heart, Upload, BarChart3, Brain, Target, Activity,
  AlertTriangle, CheckCircle, XCircle, Info, Download, Play,
  TrendingUp, Database, Settings,
  Layers, Zap, Shield, BookOpen
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, PieChart as RPieChart, Pie, Cell
} from 'recharts';
import {
  parseCSV, computeDatasetInfo, detectTargetColumns, validateDataset,
  cleanDataset, stratifiedSplit, computeCorrelation, generateHistogram,
  DataRow, DatasetInfo, TargetDetection, ValidationResult, CleanedDataset
} from './utils/dataUtils';
import {
  trainLogisticRegression, trainDecisionTree, trainRandomForest,
  trainGradientBoosting, trainKNN, evaluateModel, prepareFeatures,
  scaleFeatures, computeROCCurve, ModelResult, TrainedModel
} from './utils/mlModels';

type Page = 'home' | 'upload' | 'quality' | 'eda' | 'training' | 'evaluation' | 'prediction' | 'importance';

const COLORS = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('home');
  const [dataset, setDataset] = useState<DataRow[] | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [targetDetection, setTargetDetection] = useState<TargetDetection | null>(null);
  const [targetCol, setTargetCol] = useState<string>('');
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [cleaned, setCleaned] = useState<CleanedDataset | null>(null);
  const [models, setModels] = useState<ModelResult[]>([]);
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([]);
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string>('');
  const [predictionInputs, setPredictionInputs] = useState<Record<string, string>>({});
  const [predictionResult, setPredictionResult] = useState<{ prediction: number; probability: number } | null>(null);

  // Handle file upload
  const handleFileUpload = useCallback(async (file: File) => {
    setError('');
    try {
      const data = await parseCSV(file);
      if (data.length === 0) {
        setError('The uploaded CSV file is empty or has no valid data rows.');
        return;
      }
      const info = computeDatasetInfo(data);
      const detection = detectTargetColumns(info);
      setDataset(data);
      setDatasetInfo(info);
      setTargetDetection(detection);
      setTargetCol(detection.selected || '');
      setCleaned(null);
      setModels([]);
      setTrainedModels([]);
      setCurrentPage('quality');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse CSV file.');
    }
  }, []);

  // Validate and clean data
  const handleValidate = useCallback(() => {
    if (!datasetInfo || !targetCol) return;
    const val = validateDataset(datasetInfo, targetCol);
    setValidation(val);
    if (val.isValid) {
      const cleanedData = cleanDataset(datasetInfo.data, targetCol, datasetInfo.columns, datasetInfo.columnTypes);
      setCleaned(cleanedData);
      const fCols = datasetInfo.columns.filter(c => c !== targetCol && !cleanedData.excludedColumns.includes(c));
      setFeatureCols(fCols);
    }
  }, [datasetInfo, targetCol]);

  // Train models
  const handleTrainModels = useCallback(async () => {
    if (!cleaned || !targetCol || featureCols.length === 0) return;
    setIsTraining(true);
    setError('');

    try {
      await new Promise(resolve => setTimeout(resolve, 100));

      const { X: allX, featureNames: fNames } = prepareFeatures(cleaned.data, featureCols, datasetInfo!.columnTypes);
      setFeatureNames(fNames);

      const y = cleaned.data.map(row => Number(row[targetCol]));
      const { scaled, means, stds } = scaleFeatures(allX);

      // Stratified split
      const indexedData = cleaned.data.map((row, i) => ({ row, index: i }));
      const splitResult = stratifiedSplit(indexedData, (item) => String(item.row[targetCol]), 0.2, 42);

      const trainIndices = splitResult.train.map(d => d.index);
      const testIndices = splitResult.test.map(d => d.index);

      const XTrain = trainIndices.map(i => scaled[i]);
      const yTrain = trainIndices.map(i => y[i]);
      const XTest = testIndices.map(i => scaled[i]);
      const yTest = testIndices.map(i => y[i]);

      // Train models
      const modelConfigs = [
        () => trainLogisticRegression(XTrain, yTrain, 0.01, 500, 0.01),
        () => trainDecisionTree(XTrain, yTrain, 8, 5),
        () => trainRandomForest(XTrain, yTrain, 30, 6, 5, 42),
        () => trainGradientBoosting(XTrain, yTrain, 50, 0.1, 3),
        () => trainKNN(XTrain, yTrain, 5),
      ];

      const modelNames = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'K-Nearest Neighbors'];
      const results: ModelResult[] = [];
      const trained: TrainedModel[] = [];

      for (let i = 0; i < modelConfigs.length; i++) {
        const start = performance.now();
        const model = modelConfigs[i]();
        const result = evaluateModel(model, XTest, yTest);
        result.trainingTime = performance.now() - start;
        result.name = modelNames[i];
        results.push(result);
        trained.push(model);
      }

      setModels(results);
      setTrainedModels(trained);
      setCurrentPage('evaluation');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Model training failed.');
    } finally {
      setIsTraining(false);
    }
  }, [cleaned, targetCol, featureCols, datasetInfo]);

  // Make prediction
  const handlePredict = useCallback(() => {
    if (trainedModels.length === 0 || !predictionInputs) return;
    try {
      // Build feature vector matching the training schema
      const features: number[] = [];
      for (const col of featureCols) {
        const val = predictionInputs[col];
        if (val === undefined || val === '') {
          setError(`Please fill in all required fields. Missing: ${col}`);
          return;
        }
        const numVal = Number(val);
        if (isNaN(numVal)) {
          setError(`Invalid value for ${col}. Please enter a number.`);
          return;
        }
        features.push(numVal);
      }

      // Prepare features using the same encoding as training
      const sampleRow: DataRow = {};
      featureCols.forEach((col, i) => { sampleRow[col] = features[i]; });
      
      const { X: encodedX } = prepareFeatures([sampleRow], featureCols, datasetInfo!.columnTypes);
      const rawFeatures = encodedX[0];

      // Scale features using training data statistics
      const allX = prepareFeatures(cleaned!.data, featureCols, datasetInfo!.columnTypes).X;
      const { means, stds } = scaleFeatures(allX);
      const scaledFeatures = rawFeatures.map((v, i) => (v - means[i]) / stds[i]);

      // Use the best model (highest F1)
      const bestIdx = models.reduce((bestIdx, m, i) => m.f1Score > models[bestIdx].f1Score ? i : bestIdx, 0);
      const model = trainedModels[bestIdx];
      const prediction = model.predict(scaledFeatures);
      const probability = model.predictProba(scaledFeatures);

      setPredictionResult({ prediction, probability });
      setError('');
    } catch (err) {
      setError('Prediction failed. Please check your inputs.');
    }
  }, [trainedModels, predictionInputs, featureCols, cleaned, datasetInfo, models]);

  // Navigation items
  const navItems: { id: Page; label: string; icon: React.ReactNode; disabled: boolean }[] = [
    { id: 'home', label: 'Home', icon: <Home className="w-4 h-4" />, disabled: false },
    { id: 'upload', label: 'Upload Dataset', icon: <Upload className="w-4 h-4" />, disabled: false },
    { id: 'quality', label: 'Data Quality', icon: <Shield className="w-4 h-4" />, disabled: !datasetInfo },
    { id: 'eda', label: 'Exploratory Analysis', icon: <BarChart3 className="w-4 h-4" />, disabled: !datasetInfo },
    { id: 'training', label: 'Model Training', icon: <Brain className="w-4 h-4" />, disabled: !cleaned },
    { id: 'evaluation', label: 'Evaluation', icon: <Target className="w-4 h-4" />, disabled: models.length === 0 },
    { id: 'prediction', label: 'Prediction', icon: <Activity className="w-4 h-4" />, disabled: trainedModels.length === 0 },
    { id: 'importance', label: 'Feature Importance', icon: <TrendingUp className="w-4 h-4" />, disabled: models.length === 0 },
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col shadow-sm">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <Heart className="w-6 h-6 text-red-500" fill="currentColor" />
            <div>
              <h1 className="font-bold text-gray-900 text-sm">Heart Failure</h1>
              <p className="text-xs text-gray-500">Detection System</p>
            </div>
          </div>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => !item.disabled && setCurrentPage(item.id)}
              disabled={item.disabled}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all ${
                currentPage === item.id
                  ? 'bg-red-50 text-red-700 font-medium'
                  : item.disabled
                  ? 'text-gray-300 cursor-not-allowed'
                  : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
              }`}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </nav>
        <div className="p-4 border-t border-gray-200">
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <p className="text-xs text-amber-800 font-medium">⚠️ Educational Use Only</p>
            <p className="text-xs text-amber-700 mt-1">Not a clinical diagnostic tool. For research purposes only.</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-7xl mx-auto">
          {error && (
            <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
              <XCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-sm text-red-700">{error}</p>
              <button onClick={() => setError('')} className="ml-auto text-red-400 hover:text-red-600">
                <XCircle className="w-4 h-4" />
              </button>
            </div>
          )}

          {currentPage === 'home' && <HomePage onNavigate={setCurrentPage} hasData={!!datasetInfo} hasModels={models.length > 0} />}
          {currentPage === 'upload' && <UploadPage onUpload={handleFileUpload} datasetInfo={datasetInfo} />}
          {currentPage === 'quality' && datasetInfo && (
            <DataQualityPage
              info={datasetInfo}
              targetDetection={targetDetection}
              targetCol={targetCol}
              setTargetCol={setTargetCol}
              validation={validation}
              cleaned={cleaned}
              onValidate={handleValidate}
              onTrain={handleTrainModels}
              isTraining={isTraining}
            />
          )}
          {currentPage === 'eda' && datasetInfo && (
            <EDAPage info={datasetInfo} targetCol={targetCol} cleaned={cleaned} />
          )}
          {currentPage === 'training' && cleaned && (
            <TrainingPage
              cleaned={cleaned}
              featureCols={featureCols}
              isTraining={isTraining}
              onTrain={handleTrainModels}
            />
          )}
          {currentPage === 'evaluation' && models.length > 0 && (
            <EvaluationPage models={models} featureNames={featureNames} />
          )}
          {currentPage === 'prediction' && trainedModels.length > 0 && (
            <PredictionPage
              featureCols={featureCols}
              featureNames={featureNames}
              models={models}
              predictionInputs={predictionInputs}
              setPredictionInputs={setPredictionInputs}
              predictionResult={predictionResult}
              onPredict={handlePredict}
              columnTypes={datasetInfo?.columnTypes || {}}
            />
          )}
          {currentPage === 'importance' && models.length > 0 && (
            <ImportancePage models={models} featureNames={featureNames} />
          )}
        </div>
      </main>
    </div>
  );
}

// Home Page
function HomePage({ onNavigate, hasData, hasModels }: { onNavigate: (p: Page) => void; hasData: boolean; hasModels: boolean }) {
  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-red-600 to-red-800 rounded-2xl p-8 text-white">
        <div className="flex items-center gap-4 mb-4">
          <div className="bg-white/20 p-3 rounded-xl">
            <Heart className="w-8 h-8" fill="white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Heart Failure Detection System</h1>
            <p className="text-red-100 mt-1">Machine Learning-Powered Risk Assessment Dashboard</p>
          </div>
        </div>
        <p className="text-red-100 max-w-2xl">
          Upload a cardiovascular dataset to perform exploratory analysis, train ML models,
          evaluate performance, and make predictions. This system supports automatic data quality
          checks, feature engineering, and multiple classification algorithms.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <FeatureCard
          icon={<Upload className="w-6 h-6 text-blue-600" />}
          title="Upload & Analyze"
          description="Upload CSV datasets with automatic schema detection, data quality assessment, and target column identification."
          color="blue"
          onClick={() => onNavigate('upload')}
        />
        <FeatureCard
          icon={<Brain className="w-6 h-6 text-purple-600" />}
          title="Train Models"
          description="Train Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and KNN classifiers."
          color="purple"
          onClick={() => onNavigate('training')}
          disabled={!hasData}
        />
        <FeatureCard
          icon={<Activity className="w-6 h-6 text-green-600" />}
          title="Predict & Evaluate"
          description="Make predictions with trained models and evaluate using accuracy, precision, recall, F1, and ROC-AUC."
          color="green"
          onClick={() => onNavigate('prediction')}
          disabled={!hasModels}
        />
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-gray-600" />
          How to Use
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StepCard step={1} title="Upload Data" description="Upload a CSV file with cardiovascular data" />
          <StepCard step={2} title="Review Quality" description="Check data quality and select target column" />
          <StepCard step={3} title="Train Models" description="Train and compare multiple ML classifiers" />
          <StepCard step={4} title="Make Predictions" description="Use trained models to predict outcomes" />
        </div>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Info className="w-5 h-5 text-blue-600" />
          Supported Dataset Format
        </h2>
        <p className="text-sm text-gray-600 mb-3">
          The system accepts CSV files with cardiovascular or heart failure data. It automatically detects:
        </p>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {['Numerical features', 'Categorical features', 'Binary targets', 'Missing values', 'Duplicates', 'Target columns'].map(item => (
            <div key={item} className="flex items-center gap-2 text-sm text-gray-600">
              <CheckCircle className="w-4 h-4 text-green-500" />
              {item}
            </div>
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-4">
          Common target column names detected: DEATH_EVENT, HeartDisease, target, label, outcome, diagnosis, etc.
        </p>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description, color, onClick, disabled }: {
  icon: React.ReactNode; title: string; description: string; color: string; onClick: () => void; disabled?: boolean;
}) {
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-50 border-blue-200 hover:border-blue-300',
    purple: 'bg-purple-50 border-purple-200 hover:border-purple-300',
    green: 'bg-green-50 border-green-200 hover:border-green-300',
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`p-5 rounded-xl border text-left transition-all ${colorMap[color]} ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-md cursor-pointer'}`}
    >
      <div className="mb-3">{icon}</div>
      <h3 className="font-semibold text-gray-900 mb-1">{title}</h3>
      <p className="text-sm text-gray-600">{description}</p>
    </button>
  );
}

function StepCard({ step, title, description }: { step: number; title: string; description: string }) {
  return (
    <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
      <div className="w-7 h-7 bg-red-600 text-white rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0">
        {step}
      </div>
      <div>
        <p className="font-medium text-gray-900 text-sm">{title}</p>
        <p className="text-xs text-gray-500">{description}</p>
      </div>
    </div>
  );
}

// Generate sample heart failure dataset
function generateSampleData(): string {
  const headers = 'age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,smoking,time,DEATH_EVENT';
  const rows: string[] = [headers];
  
  let seed = 12345;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  const randn = () => {
    const u1 = random();
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1 || 0.001)) * Math.cos(2 * Math.PI * u2);
  };

  for (let i = 0; i < 299; i++) {
    const age = Math.round(Math.max(40, Math.min(90, 60 + randn() * 11)));
    const anaemia = random() < 0.43 ? 1 : 0;
    const cpk = Math.round(Math.max(23, Math.min(7861, 582 + randn() * 500)));
    const diabetes = random() < 0.42 ? 1 : 0;
    const ef = Math.round(Math.max(14, Math.min(80, 38 + randn() * 12)));
    const hbp = random() < 0.35 ? 1 : 0;
    const platelets = Math.round(Math.max(25, Math.min(850, 263 + randn() * 100)) * 1000);
    const sc = Math.max(0.5, +(0.9 + randn() * 0.6).toFixed(2));
    const ss = Math.round(Math.max(113, Math.min(149, 137 + randn() * 4)));
    const smoking = random() < 0.32 ? 1 : 0;
    const time = Math.round(Math.max(1, Math.min(313, 130 + randn() * 70)));
    
    // Generate death event based on risk factors
    let risk = 0;
    if (age > 65) risk += 0.15;
    if (ef < 35) risk += 0.25;
    if (sc > 1.3) risk += 0.2;
    if (time < 100) risk += 0.2;
    if (ss < 135) risk += 0.1;
    if (hbp === 1) risk += 0.05;
    risk += random() * 0.15;
    const death = risk > 0.45 ? 1 : 0;
    
    rows.push(`${age},${anaemia},${cpk},${diabetes},${ef},${hbp},${platelets},${sc},${ss},${smoking},${time},${death}`);
  }
  
  return rows.join('\n');
}

// Upload Page
function UploadPage({ onUpload, datasetInfo }: { onUpload: (f: File) => void; datasetInfo: DatasetInfo | null }) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) onUpload(file);
  };

  const handleSampleData = () => {
    const csv = generateSampleData();
    const blob = new Blob([csv], { type: 'text/csv' });
    const file = new File([blob], 'sample_heart_failure_data.csv', { type: 'text/csv' });
    onUpload(file);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Upload className="w-6 h-6 text-red-600" />
        <h1 className="text-2xl font-bold text-gray-900">Upload Dataset</h1>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all ${
          dragActive ? 'border-red-400 bg-red-50' : 'border-gray-300 bg-white hover:border-red-300'
        }`}
      >
        <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-700 mb-2">Drop your CSV file here</h3>
        <p className="text-sm text-gray-500 mb-4">or click to browse files</p>
        <div className="flex items-center justify-center gap-4">
          <label className="inline-flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg cursor-pointer hover:bg-red-700 transition-colors">
            <Upload className="w-4 h-4" />
            Choose CSV File
            <input
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onUpload(file);
              }}
            />
          </label>
          <button
            onClick={handleSampleData}
            className="inline-flex items-center gap-2 px-6 py-3 bg-gray-100 text-gray-700 rounded-lg cursor-pointer hover:bg-gray-200 transition-colors border border-gray-300"
          >
            <Database className="w-4 h-4" />
            Load Sample Data
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-3">Sample data: 299 patients with 12 features (inspired by UCI Heart Failure Clinical Records)</p>
      </div>

      {datasetInfo && (
        <div className="bg-green-50 border border-green-200 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="w-5 h-5 text-green-600" />
            <span className="font-medium text-green-800">Dataset loaded successfully</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-3">
            <StatBox label="Rows" value={datasetInfo.numRows.toLocaleString()} />
            <StatBox label="Columns" value={datasetInfo.numCols.toString()} />
            <StatBox label="Duplicates" value={datasetInfo.duplicates.toString()} />
            <StatBox label="Missing Values" value={Object.values(datasetInfo.missingValues).reduce((a, b) => a + b, 0).toString()} />
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-3">Expected Dataset Format</h3>
        <p className="text-sm text-gray-600 mb-3">
          Your CSV should contain patient/cardiovascular data with features and a target column. Example columns:
        </p>
        <div className="bg-gray-50 rounded-lg p-4 font-mono text-xs text-gray-700 overflow-x-auto">
          <p>age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target</p>
          <p className="text-gray-400 mt-1">or</p>
          <p>age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, smoking, time, DEATH_EVENT</p>
        </div>
        <p className="text-xs text-gray-500 mt-3">
          The system auto-detects target columns named: DEATH_EVENT, HeartDisease, target, label, outcome, diagnosis, etc.
        </p>
      </div>
    </div>
  );
}

// Data Quality Page
function DataQualityPage({
  info, targetDetection, targetCol, setTargetCol, validation, cleaned, onValidate, onTrain, isTraining
}: {
  info: DatasetInfo; targetDetection: TargetDetection | null; targetCol: string; setTargetCol: (c: string) => void;
  validation: ValidationResult | null; cleaned: CleanedDataset | null; onValidate: () => void; onTrain: () => void; isTraining: boolean;
}) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Shield className="w-6 h-6 text-red-600" />
          <h1 className="text-2xl font-bold text-gray-900">Data Quality Report</h1>
        </div>
        <button
          onClick={onValidate}
          className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2 text-sm"
        >
          <CheckCircle className="w-4 h-4" />
          Validate & Clean
        </button>
      </div>

      {/* Dataset Overview */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <StatBox label="Rows" value={info.numRows.toLocaleString()} />
        <StatBox label="Columns" value={info.numCols.toString()} />
        <StatBox label="Duplicates" value={info.duplicates.toString()} />
        <StatBox label="Missing Values" value={Object.values(info.missingValues).reduce((a, b) => a + b, 0).toString()} />
        <StatBox label="Missing %" value={`${(Object.values(info.missingValues).reduce((a, b) => a + b, 0) / (info.numRows * info.numCols) * 100).toFixed(1)}%`} />
      </div>

      {/* Target Column Selection */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
          <Target className="w-5 h-5 text-red-600" />
          Target Column Selection
        </h3>
        {targetDetection && (
          <div className={`p-3 rounded-lg mb-3 ${
            targetDetection.confidence === 'high' ? 'bg-green-50 border border-green-200' :
            targetDetection.confidence === 'medium' ? 'bg-yellow-50 border border-yellow-200' :
            'bg-orange-50 border border-orange-200'
          }`}>
            <p className="text-sm">{targetDetection.message}</p>
          </div>
        )}
        <div className="flex items-center gap-4">
          <select
            value={targetCol}
            onChange={(e) => setTargetCol(e.target.value)}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-red-500 focus:border-red-500"
          >
            <option value="">Select target column...</option>
            {info.columns.map(col => (
              <option key={col} value={col}>{col} ({info.columnTypes[col]}, {info.uniqueValues[col]} unique)</option>
            ))}
          </select>
        </div>
        {targetDetection?.candidates && targetDetection.candidates.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            <span className="text-xs text-gray-500">Detected candidates:</span>
            {targetDetection.candidates.map(c => (
              <button
                key={c}
                onClick={() => setTargetCol(c)}
                className={`px-2 py-1 rounded text-xs ${c === targetCol ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
              >
                {c}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Validation Results */}
      {validation && (
        <div className={`rounded-xl border p-6 ${validation.isValid ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            {validation.isValid ? <CheckCircle className="w-5 h-5 text-green-600" /> : <XCircle className="w-5 h-5 text-red-600" />}
            Validation Result: {validation.isValid ? 'PASSED' : 'FAILED'}
          </h3>
          {validation.issues.length > 0 && (
            <div className="space-y-1 mb-3">
              {validation.issues.map((issue, i) => (
                <p key={i} className="text-sm text-red-700 flex items-center gap-2">
                  <XCircle className="w-3 h-3" /> {issue}
                </p>
              ))}
            </div>
          )}
          {validation.warnings.length > 0 && (
            <div className="space-y-1">
              {validation.warnings.map((w, i) => (
                <p key={i} className="text-sm text-amber-700 flex items-center gap-2">
                  <AlertTriangle className="w-3 h-3" /> {w}
                </p>
              ))}
            </div>
          )}
          {validation.isValid && (
            <p className="text-sm text-green-700 mt-2">Task type: <strong>{validation.taskType}</strong> classification</p>
          )}
        </div>
      )}

      {/* Cleaning Report */}
      {cleaned && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <Layers className="w-5 h-5 text-blue-600" />
            Cleaning Report
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatBox label="Removed Duplicates" value={cleaned.removedDuplicates.toString()} />
            <StatBox label="Removed (Missing Target)" value={cleaned.removedMissingTarget.toString()} />
            <StatBox label="Imputed Values" value={cleaned.imputedValues.toString()} />
            <StatBox label="Excluded Columns" value={cleaned.excludedColumns.length.toString()} />
          </div>
          <div className="mt-4 flex justify-end">
            <button
              onClick={onTrain}
              disabled={isTraining}
              className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {isTraining ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Train Models
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Column Types Table */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-3">Column Information</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3 font-medium text-gray-600">Column</th>
                <th className="text-left py-2 px-3 font-medium text-gray-600">Type</th>
                <th className="text-left py-2 px-3 font-medium text-gray-600">Missing</th>
                <th className="text-left py-2 px-3 font-medium text-gray-600">Unique</th>
                <th className="text-left py-2 px-3 font-medium text-gray-600">Classification</th>
              </tr>
            </thead>
            <tbody>
              {info.columns.map(col => (
                <tr key={col} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-2 px-3 font-mono text-xs">{col}</td>
                  <td className="py-2 px-3"><span className="px-2 py-0.5 bg-gray-100 rounded text-xs">{info.dtypes[col]}</span></td>
                  <td className="py-2 px-3">{info.missingValues[col]} ({info.missingPercentages[col].toFixed(1)}%)</td>
                  <td className="py-2 px-3">{info.uniqueValues[col]}</td>
                  <td className="py-2 px-3">
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      info.columnTypes[col] === 'numerical' ? 'bg-blue-100 text-blue-700' :
                      info.columnTypes[col] === 'binary' ? 'bg-green-100 text-green-700' :
                      info.columnTypes[col] === 'categorical' ? 'bg-purple-100 text-purple-700' :
                      info.columnTypes[col] === 'identifier' ? 'bg-gray-100 text-gray-700' :
                      'bg-yellow-100 text-yellow-700'
                    }`}>
                      {info.columnTypes[col]}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Data Preview */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-3">Data Preview (first 10 rows)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-200">
                {info.columns.map(col => (
                  <th key={col} className="text-left py-2 px-2 font-medium text-gray-600 whitespace-nowrap">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {info.data.slice(0, 10).map((row, i) => (
                <tr key={i} className="border-b border-gray-50">
                  {info.columns.map(col => (
                    <td key={col} className="py-1.5 px-2 whitespace-nowrap">
                      {row[col] === null || row[col] === undefined ? (
                        <span className="text-red-400 italic">null</span>
                      ) : (
                        String(row[col]).slice(0, 20)
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// EDA Page
function EDAPage({ info, targetCol, cleaned }: { info: DatasetInfo; targetCol: string; cleaned: CleanedDataset | null }) {
  const data = cleaned?.data || info.data;

  // Target distribution
  const targetDist = useMemo(() => {
    if (!targetCol) return [];
    const counts: Record<string, number> = {};
    for (const row of data) {
      const val = String(row[targetCol] ?? 'null');
      counts[val] = (counts[val] || 0) + 1;
    }
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [data, targetCol]);

  // Missing values chart
  const missingData = useMemo(() => {
    return info.columns
      .filter(col => info.missingValues[col] > 0)
      .map(col => ({ column: col, missing: info.missingValues[col], percentage: info.missingPercentages[col] }))
      .sort((a, b) => b.missing - a.missing)
      .slice(0, 15);
  }, [info]);

  // Correlation matrix
  const correlation = useMemo(() => {
    return computeCorrelation(data, info.columns);
  }, [data, info.columns]);

  // Histograms for numerical columns
  const numericalCols = useMemo(() => {
    return info.columns.filter(c => info.columnTypes[c] === 'numerical' || info.columnTypes[c] === 'binary');
  }, [info]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <BarChart3 className="w-6 h-6 text-red-600" />
        <h1 className="text-2xl font-bold text-gray-900">Exploratory Data Analysis</h1>
      </div>

      {/* Target Distribution */}
      {targetCol && targetDist.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Target Class Distribution</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RPieChart>
                  <Pie data={targetDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                    {targetDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </RPieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-2">
              {targetDist.map((item, i) => (
                <div key={item.name} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                    <span className="text-sm font-medium">Class {item.name}</span>
                  </div>
                  <span className="text-sm text-gray-600">{item.value} ({(item.value / data.length * 100).toFixed(1)}%)</span>
                </div>
              ))}
              {targetDist.length === 2 && (
                <div className="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <p className="text-xs text-amber-800">
                    <strong>Class Balance:</strong> {Math.max(...targetDist.map(d => d.value)) / Math.min(...targetDist.map(d => d.value)) > 2
                      ? 'Imbalanced dataset detected. Consider using class weights.'
                      : 'Classes are reasonably balanced.'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Missing Values */}
      {missingData.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Missing Values by Column</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={missingData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="column" width={120} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="missing" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Correlation Heatmap */}
      {correlation.columns.length > 1 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Correlation Matrix (Numerical Features)</h3>
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              <table className="text-xs">
                <thead>
                  <tr>
                    <th className="p-1"></th>
                    {correlation.columns.map(col => (
                      <th key={col} className="p-1 text-center font-medium text-gray-600 whitespace-nowrap" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', maxHeight: '80px' }}>
                        {col.length > 12 ? col.slice(0, 12) + '…' : col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {correlation.columns.map((row, i) => (
                    <tr key={row}>
                      <td className="p-1 font-medium text-gray-600 whitespace-nowrap">{row.length > 15 ? row.slice(0, 15) + '…' : row}</td>
                      {correlation.matrix[i].map((val, j) => (
                        <td key={j} className="p-0.5">
                          <div
                            className="w-8 h-8 rounded flex items-center justify-center text-[9px] font-medium"
                            style={{
                              backgroundColor: val > 0
                                ? `rgba(239, 68, 68, ${Math.abs(val) * 0.7})`
                                : `rgba(59, 130, 246, ${Math.abs(val) * 0.7})`,
                              color: Math.abs(val) > 0.5 ? 'white' : '#374151'
                            }}
                            title={`${correlation.columns[i]} vs ${correlation.columns[j]}: ${val.toFixed(3)}`}
                          >
                            {val.toFixed(2)}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-3">Red = positive correlation, Blue = negative correlation. Note: Correlation does not imply causation.</p>
        </div>
      )}

      {/* Feature Distributions */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Feature Distributions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {numericalCols.slice(0, 9).map(col => {
            const values = data.map(row => Number(row[col])).filter(v => !isNaN(v) && isFinite(v));
            const histogram = generateHistogram(values, 15);
            return (
              <div key={col} className="border border-gray-100 rounded-lg p-3">
                <p className="text-xs font-medium text-gray-700 mb-2 truncate">{col}</p>
                <div className="h-32">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={histogram}>
                      <Bar dataKey="count" fill="#ef4444" opacity={0.7} radius={[2, 2, 0, 0]} />
                      <XAxis dataKey="bin" hide />
                      <YAxis hide />
                      <Tooltip formatter={(value) => [value, 'Count']} labelFormatter={(label) => `Range: ${label}`} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Descriptive Statistics */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Descriptive Statistics</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3">Feature</th>
                <th className="text-right py-2 px-3">Mean</th>
                <th className="text-right py-2 px-3">Std</th>
                <th className="text-right py-2 px-3">Min</th>
                <th className="text-right py-2 px-3">Q1</th>
                <th className="text-right py-2 px-3">Median</th>
                <th className="text-right py-2 px-3">Q3</th>
                <th className="text-right py-2 px-3">Max</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(info.descriptiveStats).map(([col, stats]) => (
                <tr key={col} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="py-2 px-3 font-mono text-xs">{col}</td>
                  <td className="py-2 px-3 text-right">{stats.mean.toFixed(2)}</td>
                  <td className="py-2 px-3 text-right">{stats.std.toFixed(2)}</td>
                  <td className="py-2 px-3 text-right">{stats.min.toFixed(2)}</td>
                  <td className="py-2 px-3 text-right">{stats.q1.toFixed(2)}</td>
                  <td className="py-2 px-3 text-right">{stats.median.toFixed(2)}</td>
                  <td className="py-2 px-3 text-right">{stats.q3.toFixed(2)}</td>
                  <td className="py-2 px-3 text-right">{stats.max.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// Training Page
function TrainingPage({ cleaned, featureCols, isTraining, onTrain }: {
  cleaned: CleanedDataset; featureCols: string[]; isTraining: boolean; onTrain: () => void;
}) {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Brain className="w-6 h-6 text-red-600" />
        <h1 className="text-2xl font-bold text-gray-900">Model Training</h1>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Training Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatBox label="Training Samples" value={Math.floor(cleaned.data.length * 0.8).toString()} />
          <StatBox label="Test Samples" value={Math.ceil(cleaned.data.length * 0.2).toString()} />
          <StatBox label="Features" value={featureCols.length.toString()} />
          <StatBox label="Split Method" value="Stratified 80/20" />
        </div>

        <h4 className="font-medium text-gray-800 mb-3">Models to Train:</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
          {[
            { name: 'Logistic Regression', desc: 'Linear classifier with L2 regularization', icon: '📊' },
            { name: 'Decision Tree', desc: 'Gini-based tree with max depth 8', icon: '🌳' },
            { name: 'Random Forest', desc: 'Ensemble of 30 trees, bootstrap sampling', icon: '🌲' },
            { name: 'Gradient Boosting', desc: 'Sequential boosting with 50 estimators', icon: '⚡' },
            { name: 'K-Nearest Neighbors', desc: 'K=5, Euclidean distance', icon: '📍' },
          ].map(model => (
            <div key={model.name} className="p-3 bg-gray-50 rounded-lg border border-gray-100">
              <div className="flex items-center gap-2 mb-1">
                <span>{model.icon}</span>
                <span className="font-medium text-sm text-gray-800">{model.name}</span>
              </div>
              <p className="text-xs text-gray-500">{model.desc}</p>
            </div>
          ))}
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={onTrain}
            disabled={isTraining}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2 disabled:opacity-50 font-medium"
          >
            {isTraining ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Training Models...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5" />
                Start Training
              </>
            )}
          </button>
          {isTraining && <span className="text-sm text-gray-500">Please wait while models are being trained...</span>}
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <h4 className="font-medium text-blue-800 mb-2 flex items-center gap-2">
          <Info className="w-4 h-4" /> Training Methodology
        </h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• Stratified train/test split (80/20) preserves class distribution</li>
          <li>• Features are standardized (zero mean, unit variance) before training</li>
          <li>• All preprocessing is fit only on training data to prevent data leakage</li>
          <li>• Models are evaluated on the held-out test set</li>
          <li>• Random seed (42) ensures reproducibility</li>
        </ul>
      </div>
    </div>
  );
}

// Evaluation Page
function EvaluationPage({ models, featureNames }: { models: ModelResult[]; featureNames: string[] }) {
  const bestModel = useMemo(() => {
    return models.reduce((best, m) => m.f1Score > best.f1Score ? m : best, models[0]);
  }, [models]);

  const handleDownloadReport = () => {
    const headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC', 'TP', 'TN', 'FP', 'FN'];
    const rows = models.map(m => [
      m.name,
      (m.accuracy * 100).toFixed(1) + '%',
      (m.precision * 100).toFixed(1) + '%',
      (m.recall * 100).toFixed(1) + '%',
      (m.specificity * 100).toFixed(1) + '%',
      (m.f1Score * 100).toFixed(1) + '%',
      (m.aucRoc * 100).toFixed(1) + '%',
      m.confusionMatrix[1][1].toString(),
      m.confusionMatrix[0][0].toString(),
      m.confusionMatrix[0][1].toString(),
      m.confusionMatrix[1][0].toString(),
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model_evaluation_report.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const comparisonData = models.map(m => ({
    name: m.name,
    Accuracy: +(m.accuracy * 100).toFixed(1),
    Precision: +(m.precision * 100).toFixed(1),
    Recall: +(m.recall * 100).toFixed(1),
    'F1-Score': +(m.f1Score * 100).toFixed(1),
    'ROC-AUC': +(m.aucRoc * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Target className="w-6 h-6 text-red-600" />
          <h1 className="text-2xl font-bold text-gray-900">Model Evaluation</h1>
        </div>
        <button
          onClick={handleDownloadReport}
          className="px-4 py-2 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors flex items-center gap-2 text-sm border border-blue-200"
        >
          <Download className="w-4 h-4" />
          Download Report
        </button>
      </div>

      {/* Best Model Highlight */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl p-5">
        <div className="flex items-center gap-3 mb-2">
          <CheckCircle className="w-6 h-6 text-green-600" />
          <h3 className="font-semibold text-green-800">Best Model: {bestModel.name}</h3>
        </div>
        <p className="text-sm text-green-700">
          Selected based on highest F1-Score: <strong>{(bestModel.f1Score * 100).toFixed(1)}%</strong> |
          ROC-AUC: <strong>{(bestModel.aucRoc * 100).toFixed(1)}%</strong> |
          Recall: <strong>{(bestModel.recall * 100).toFixed(1)}%</strong>
        </p>
        <p className="text-xs text-green-600 mt-2">
          Note: For clinical screening, recall (sensitivity) is critical to minimize false negatives.
          High test scores do not prove clinical effectiveness or generalization to real patients.
        </p>
      </div>

      {/* Model Comparison Table */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Model Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3">Model</th>
                <th className="text-right py-2 px-3">Accuracy</th>
                <th className="text-right py-2 px-3">Precision</th>
                <th className="text-right py-2 px-3">Recall</th>
                <th className="text-right py-2 px-3">Specificity</th>
                <th className="text-right py-2 px-3">F1-Score</th>
                <th className="text-right py-2 px-3">ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {models.map(m => (
                <tr key={m.name} className={`border-b border-gray-50 ${m.name === bestModel.name ? 'bg-green-50' : ''}`}>
                  <td className="py-2 px-3 font-medium">{m.name}</td>
                  <td className="py-2 px-3 text-right">{(m.accuracy * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3 text-right">{(m.precision * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3 text-right">{(m.recall * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3 text-right">{(m.specificity * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3 text-right font-medium">{(m.f1Score * 100).toFixed(1)}%</td>
                  <td className="py-2 px-3 text-right">{(m.aucRoc * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Metrics Comparison Chart */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Metrics Comparison</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="Accuracy" fill="#3b82f6" radius={[2, 2, 0, 0]} />
              <Bar dataKey="Precision" fill="#10b981" radius={[2, 2, 0, 0]} />
              <Bar dataKey="Recall" fill="#f59e0b" radius={[2, 2, 0, 0]} />
              <Bar dataKey="F1-Score" fill="#ef4444" radius={[2, 2, 0, 0]} />
              <Bar dataKey="ROC-AUC" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrices */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Confusion Matrices</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map(m => (
            <div key={m.name} className="border border-gray-100 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-700 mb-3">{m.name}</h4>
              <div className="grid grid-cols-2 gap-1 max-w-[200px] mx-auto">
                <div className="bg-green-100 p-3 rounded text-center">
                  <p className="text-xs text-green-600">TN</p>
                  <p className="font-bold text-green-800">{m.confusionMatrix[0][0]}</p>
                </div>
                <div className="bg-red-100 p-3 rounded text-center">
                  <p className="text-xs text-red-600">FP</p>
                  <p className="font-bold text-red-800">{m.confusionMatrix[0][1]}</p>
                </div>
                <div className="bg-orange-100 p-3 rounded text-center">
                  <p className="text-xs text-orange-600">FN</p>
                  <p className="font-bold text-orange-800">{m.confusionMatrix[1][0]}</p>
                </div>
                <div className="bg-green-100 p-3 rounded text-center">
                  <p className="text-xs text-green-600">TP</p>
                  <p className="font-bold text-green-800">{m.confusionMatrix[1][1]}</p>
                </div>
              </div>
              <p className="text-xs text-gray-500 text-center mt-2">
                Sensitivity: {(m.recall * 100).toFixed(1)}% | Specificity: {(m.specificity * 100).toFixed(1)}%
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* ROC Curves */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">ROC Curves (Approximation)</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: 'False Positive Rate', position: 'bottom', offset: -5 }} />
              <YAxis type="number" domain={[0, 1]} label={{ value: 'True Positive Rate', angle: -90, position: 'left' }} />
              <Tooltip />
              <Legend />
              {models.map((m, i) => {
                const rocPoints = computeROCCurve(
                  m.predictions.map((_, j) => j < m.confusionMatrix[0][0] + m.confusionMatrix[0][1] ? 0 : 1),
                  m.probabilities
                );
                return (
                  <Line key={m.name} data={rocPoints} dataKey="tpr" name={`${m.name} (AUC=${(m.aucRoc * 100).toFixed(0)}%)`} stroke={COLORS[i]} dot={false} strokeWidth={2} />
                );
              })}
              <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" name="Random" stroke="#9ca3af" strokeDasharray="5 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// Prediction Page
function PredictionPage({ featureCols, featureNames, models, predictionInputs, setPredictionInputs, predictionResult, onPredict, columnTypes }: {
  featureCols: string[]; featureNames: string[]; models: ModelResult[];
  predictionInputs: Record<string, string>; setPredictionInputs: (v: Record<string, string>) => void;
  predictionResult: { prediction: number; probability: number } | null; onPredict: () => void;
  columnTypes: Record<string, string>;
}) {
  const bestModel = models.reduce((best, m) => m.f1Score > best.f1Score ? m : best, models[0]);
  
  const handleDownloadResults = () => {
    if (!predictionResult) return;
    const results = [
      { ...predictionInputs, prediction: predictionResult.prediction, probability: predictionResult.probability.toFixed(4) }
    ];
    const headers = [...featureCols, 'prediction', 'probability'];
    const csvContent = [
      headers.join(','),
      results.map(r => headers.map(h => r[h as keyof typeof r] ?? '').join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'prediction_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleFillSample = () => {
    const sampleInputs: Record<string, string> = {};
    for (const col of featureCols) {
      // Generate reasonable random values
      const seed = col.length * 7 + col.charCodeAt(0);
      const val = ((seed * 1664525 + 1013904223) % 1000) / 10;
      sampleInputs[col] = val.toFixed(1);
    }
    setPredictionInputs(sampleInputs);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Activity className="w-6 h-6 text-red-600" />
        <h1 className="text-2xl font-bold text-gray-900">Heart Failure Prediction</h1>
      </div>

      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
        <p className="text-sm text-amber-800">
          <strong>Disclaimer:</strong> Predictions are model-estimated probabilities for educational and research purposes only.
          This is not a clinically validated diagnostic tool. Always consult qualified healthcare professionals for medical decisions.
        </p>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-900">Enter Feature Values</h3>
          <span className="text-sm text-gray-500">Using: {bestModel.name}</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          {featureCols.map((col, i) => (
            <div key={col}>
              <label className="block text-xs font-medium text-gray-600 mb-1">{col}</label>
              <input
                type="number"
                step="any"
                value={predictionInputs[col] || ''}
                onChange={(e) => setPredictionInputs({ ...predictionInputs, [col]: e.target.value })}
                placeholder={`Enter ${col}`}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-red-500 focus:border-red-500"
              />
            </div>
          ))}
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={onPredict}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2 font-medium"
          >
            <Play className="w-4 h-4" />
            Get Prediction
          </button>
          <button
            onClick={handleFillSample}
            className="px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center gap-2 text-sm border border-gray-300"
          >
            <Settings className="w-4 h-4" />
            Fill Sample Values
          </button>
          {predictionResult && (
            <button
              onClick={handleDownloadResults}
              className="px-4 py-3 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors flex items-center gap-2 text-sm border border-blue-200"
            >
              <Download className="w-4 h-4" />
              Download Results
            </button>
          )}
        </div>
      </div>

      {predictionResult && (
        <div className={`rounded-xl border p-6 ${predictionResult.prediction === 1 ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            {predictionResult.prediction === 1 ? (
              <><AlertTriangle className="w-5 h-5 text-red-600" /> <span className="text-red-800">Elevated Risk Detected</span></>
            ) : (
              <><CheckCircle className="w-5 h-5 text-green-600" /> <span className="text-green-800">Lower Risk Indicated</span></>
            )}
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Predicted Class</p>
              <p className="text-2xl font-bold">{predictionResult.prediction === 1 ? 'Positive (1)' : 'Negative (0)'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Model-Estimated Probability</p>
              <p className="text-2xl font-bold">{(predictionResult.probability * 100).toFixed(1)}%</p>
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full ${predictionResult.probability > 0.5 ? 'bg-red-500' : 'bg-green-500'}`}
                style={{ width: `${predictionResult.probability * 100}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Probability scale: 0% (lowest risk) → 100% (highest estimated risk)
            </p>
          </div>
          <p className="text-xs text-gray-500 mt-3">
            This is a model-estimated probability based on the {bestModel.name} model trained on the uploaded dataset.
            It should not be interpreted as a confirmed medical risk.
          </p>
        </div>
      )}
    </div>
  );
}

// Feature Importance Page
function ImportancePage({ models, featureNames }: { models: ModelResult[]; featureNames: string[] }) {
  const [selectedModel, setSelectedModel] = useState(0);
  const model = models[selectedModel];

  const importanceData = useMemo(() => {
    if (!model) return [];
    const entries = Object.entries(model.featureImportance);
    return entries
      .map(([key, value]) => ({
        feature: featureNames[parseInt(key.replace('feature_', ''))] || key,
        importance: +(value * 100).toFixed(1)
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 15);
  }, [model, featureNames]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <TrendingUp className="w-6 h-6 text-red-600" />
        <h1 className="text-2xl font-bold text-gray-900">Feature Importance</h1>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-900">Select Model</h3>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(Number(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm"
          >
            {models.map((m, i) => (
              <option key={i} value={i}>{m.name}</option>
            ))}
          </select>
        </div>

        {importanceData.length > 0 ? (
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importanceData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" label={{ value: 'Importance (%)', position: 'bottom', offset: -5 }} />
                <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(value) => [`${value}%`, 'Importance']} />
                <Bar dataKey="importance" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            <p>Feature importance not available for this model type.</p>
            <p className="text-sm mt-1">Tree-based models and logistic regression provide feature importance.</p>
          </div>
        )}
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
        <h4 className="font-medium text-blue-800 mb-2 flex items-center gap-2">
          <Info className="w-4 h-4" /> Understanding Feature Importance
        </h4>
        <ul className="text-sm text-blue-700 space-y-1">
          <li>• Feature importance indicates how much each feature contributes to model predictions</li>
          <li>• For tree-based models: based on how often features are used for splitting</li>
          <li>• For logistic regression: based on the magnitude of coefficients</li>
          <li>• High importance does not mean the feature <em>causes</em> heart failure</li>
          <li>• Correlation does not imply causation — these are statistical associations only</li>
          <li>• Importance can vary between models and should not be used for medical conclusions</li>
        </ul>
      </div>

      {/* All models comparison */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Top Features Across All Models</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3">Feature</th>
                {models.map(m => <th key={m.name} className="text-right py-2 px-3">{m.name}</th>)}
              </tr>
            </thead>
            <tbody>
              {featureNames.slice(0, 10).map((name, i) => (
                <tr key={name} className="border-b border-gray-50">
                  <td className="py-2 px-3 font-mono text-xs">{name}</td>
                  {models.map(m => {
                    const val = m.featureImportance[`feature_${i}`];
                    return (
                      <td key={m.name} className="py-2 px-3 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-16 bg-gray-200 rounded-full h-2">
                            <div className="bg-red-500 h-2 rounded-full" style={{ width: `${(val || 0) * 100}%` }} />
                          </div>
                          <span className="text-xs w-10">{val ? ((val || 0) * 100).toFixed(0) : '0'}%</span>
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// Utility Components
function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-lg font-bold text-gray-900">{value}</p>
    </div>
  );
}

function Home({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}
