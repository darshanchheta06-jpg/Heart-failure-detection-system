import Papa from 'papaparse';

// Types
export interface DataRow {
  [key: string]: string | number | null;
}

export interface DatasetInfo {
  data: DataRow[];
  columns: string[];
  numRows: number;
  numCols: number;
  dtypes: Record<string, string>;
  missingValues: Record<string, number>;
  missingPercentages: Record<string, number>;
  uniqueValues: Record<string, number>;
  descriptiveStats: Record<string, { mean: number; std: number; min: number; max: number; median: number; q1: number; q3: number }>;
  duplicates: number;
  columnTypes: Record<string, 'numerical' | 'categorical' | 'binary' | 'identifier' | 'date'>;
}

export interface TargetDetection {
  candidates: string[];
  selected: string | null;
  confidence: 'high' | 'medium' | 'low' | 'none';
  message: string;
}

export interface CleanedDataset {
  data: DataRow[];
  originalData: DataRow[];
  removedDuplicates: number;
  removedMissingTarget: number;
  imputedValues: number;
  excludedColumns: string[];
}

export interface ValidationResult {
  isValid: boolean;
  issues: string[];
  warnings: string[];
  taskType: 'binary' | 'multiclass' | 'regression' | 'unknown';
}

// Known target column names for heart failure detection
const TARGET_COLUMN_NAMES = [
  'death_event', 'heartdisease', 'heart_failure', 'target', 'label',
  'outcome', 'diagnosis', 'class', 'result', 'status', 'deathevent',
  'cardiovascular', 'cvd', 'chd', 'hf', 'failure', 'event',
  '10_year_chd', 'tenyearchd', 'heart_attack', 'stroke'
];

// Parse CSV file
export function parseCSV(file: File): Promise<DataRow[]> {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      complete: (results) => {
        if (results.errors.length > 0) {
          console.warn('CSV parsing warnings:', results.errors);
        }
        resolve(results.data as DataRow[]);
      },
      error: (error) => {
        reject(new Error(`CSV parsing failed: ${error.message}`));
      }
    });
  });
}

// Detect column types
export function detectColumnTypes(data: DataRow[], columns: string[]): Record<string, string> {
  const dtypes: Record<string, string> = {};
  
  for (const col of columns) {
    const values = data.map(row => row[col]).filter(v => v !== null && v !== undefined && v !== '');
    if (values.length === 0) {
      dtypes[col] = 'empty';
      continue;
    }
    
    const numericCount = values.filter(v => typeof v === 'number' && !isNaN(v)).length;
    const ratio = numericCount / values.length;
    
    if (ratio > 0.8) {
      dtypes[col] = 'float64';
    } else {
      const uniqueVals = new Set(values.map(v => String(v).toLowerCase()));
      if (uniqueVals.size <= 2) {
        dtypes[col] = 'binary';
      } else if (uniqueVals.size <= 20) {
        dtypes[col] = 'category';
      } else {
        dtypes[col] = 'object';
      }
    }
  }
  
  return dtypes;
}

// Classify columns into types
export function classifyColumns(data: DataRow[], columns: string[], dtypes: Record<string, string>): Record<string, 'numerical' | 'categorical' | 'binary' | 'identifier' | 'date'> {
  const types: Record<string, 'numerical' | 'categorical' | 'binary' | 'identifier' | 'date'> = {};
  
  for (const col of columns) {
    const values = data.map(row => row[col]).filter(v => v !== null && v !== undefined && v !== '');
    const uniqueCount = new Set(values.map(v => String(v))).size;
    
    // Check for identifier columns
    if (uniqueCount === values.length && values.length > 10) {
      types[col] = 'identifier';
      continue;
    }
    
    // Check for date columns
    const datePattern = /^\d{4}[-/]\d{1,2}[-/]\d{1,2}/;
    const dateCount = values.filter(v => datePattern.test(String(v))).length;
    if (dateCount > values.length * 0.8) {
      types[col] = 'date';
      continue;
    }
    
    // Check for binary
    if (uniqueCount <= 2 && values.length > 0) {
      types[col] = 'binary';
      continue;
    }
    
    // Check for numerical
    if (dtypes[col] === 'float64') {
      types[col] = 'numerical';
      continue;
    }
    
    // Default to categorical
    types[col] = 'categorical';
  }
  
  return types;
}

// Compute dataset information
export function computeDatasetInfo(data: DataRow[]): DatasetInfo {
  if (data.length === 0) {
    throw new Error('Dataset is empty');
  }
  
  const columns = Object.keys(data[0]);
  const numRows = data.length;
  const numCols = columns.length;
  const dtypes = detectColumnTypes(data, columns);
  const columnTypes = classifyColumns(data, columns, dtypes);
  
  // Missing values
  const missingValues: Record<string, number> = {};
  const missingPercentages: Record<string, number> = {};
  const uniqueValues: Record<string, number> = {};
  
  for (const col of columns) {
    const missing = data.filter(row => row[col] === null || row[col] === undefined || row[col] === '').length;
    missingValues[col] = missing;
    missingPercentages[col] = (missing / numRows) * 100;
    uniqueValues[col] = new Set(data.map(row => row[col]).filter(v => v !== null && v !== undefined && v !== '')).size;
  }
  
  // Descriptive statistics for numerical columns
  const descriptiveStats: Record<string, { mean: number; std: number; min: number; max: number; median: number; q1: number; q3: number }> = {};
  
  for (const col of columns) {
    if (columnTypes[col] === 'numerical' || columnTypes[col] === 'binary') {
      const values = data.map(row => Number(row[col])).filter(v => !isNaN(v) && isFinite(v));
      if (values.length > 0) {
        const sorted = [...values].sort((a, b) => a - b);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        const min = sorted[0];
        const max = sorted[sorted.length - 1];
        const median = sorted.length % 2 === 0 
          ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2 
          : sorted[Math.floor(sorted.length / 2)];
        const q1 = sorted[Math.floor(sorted.length * 0.25)];
        const q3 = sorted[Math.floor(sorted.length * 0.75)];
        
        descriptiveStats[col] = { mean, std, min, max, median, q1, q3 };
      }
    }
  }
  
  // Count duplicates
  const seen = new Set<string>();
  let duplicates = 0;
  for (const row of data) {
    const key = JSON.stringify(row);
    if (seen.has(key)) {
      duplicates++;
    } else {
      seen.add(key);
    }
  }
  
  return {
    data,
    columns,
    numRows,
    numCols,
    dtypes,
    missingValues,
    missingPercentages,
    uniqueValues,
    descriptiveStats,
    duplicates,
    columnTypes
  };
}

// Detect potential target columns
export function detectTargetColumns(info: DatasetInfo): TargetDetection {
  const candidates: string[] = [];
  
  for (const col of info.columns) {
    const colLower = col.toLowerCase().replace(/[_\s-]/g, '');
    const isTargetName = TARGET_COLUMN_NAMES.some(name => colLower.includes(name.replace(/[_\s-]/g, '')));
    const isBinary = info.columnTypes[col] === 'binary';
    const uniqueCount = info.uniqueValues[col];
    
    if (isTargetName && (isBinary || uniqueCount <= 5)) {
      candidates.push(col);
    } else if (isBinary && uniqueCount === 2) {
      candidates.push(col);
    }
  }
  
  let confidence: 'high' | 'medium' | 'low' | 'none' = 'none';
  let message = '';
  let selected: string | null = null;
  
  if (candidates.length === 1) {
    confidence = 'high';
    selected = candidates[0];
    message = `Automatically detected "${candidates[0]}" as the likely target column.`;
  } else if (candidates.length > 1) {
    confidence = 'medium';
    selected = candidates[0];
    message = `Multiple potential target columns found: ${candidates.join(', ')}. Please confirm your selection.`;
  } else {
    confidence = 'none';
    message = 'No obvious target column detected. Please manually select the target column.';
  }
  
  return { candidates, selected, confidence, message };
}

// Validate dataset for ML task
export function validateDataset(info: DatasetInfo, targetCol: string): ValidationResult {
  const issues: string[] = [];
  const warnings: string[] = [];
  let taskType: 'binary' | 'multiclass' | 'regression' | 'unknown' = 'unknown';
  
  if (!targetCol) {
    issues.push('No target column selected.');
    return { isValid: false, issues, warnings, taskType };
  }
  
  if (!info.columns.includes(targetCol)) {
    issues.push(`Target column "${targetCol}" not found in dataset.`);
    return { isValid: false, issues, warnings, taskType };
  }
  
  const targetValues = info.data.map(row => row[targetCol]).filter(v => v !== null && v !== undefined && v !== '');
  const uniqueTargetValues = new Set(targetValues.map(v => String(v)));
  const numClasses = uniqueTargetValues.size;
  
  if (targetValues.length === 0) {
    issues.push('Target column has no valid values.');
    return { isValid: false, issues, warnings, taskType };
  }
  
  if (numClasses === 1) {
    issues.push(`Target column has only one class: "${[...uniqueTargetValues][0]}". Classification requires at least two classes.`);
    return { isValid: false, issues, warnings, taskType };
  }
  
  if (numClasses === 2) {
    taskType = 'binary';
  } else if (numClasses <= 10) {
    taskType = 'multiclass';
  } else if (info.columnTypes[targetCol] === 'numerical') {
    taskType = 'regression';
    issues.push('Target appears to be continuous. This system is designed for classification tasks.');
  } else {
    taskType = 'multiclass';
  }
  
  // Check class distribution
  const classCounts: Record<string, number> = {};
  for (const v of targetValues) {
    const key = String(v);
    classCounts[key] = (classCounts[key] || 0) + 1;
  }
  
  const minClassCount = Math.min(...Object.values(classCounts));
  const maxClassCount = Math.max(...Object.values(classCounts));
  const imbalanceRatio = maxClassCount / minClassCount;
  
  if (imbalanceRatio > 10) {
    warnings.push(`Severe class imbalance detected (ratio ${imbalanceRatio.toFixed(1)}:1). Consider using class weights or resampling.`);
  } else if (imbalanceRatio > 3) {
    warnings.push(`Moderate class imbalance detected (ratio ${imbalanceRatio.toFixed(1)}:1).`);
  }
  
  if (minClassCount < 10) {
    warnings.push(`Smallest class has only ${minClassCount} samples. Results may be unreliable.`);
  }
  
  if (info.numRows < 50) {
    warnings.push(`Dataset has only ${info.numRows} rows. Results may be unreliable with small datasets.`);
  }
  
  // Check feature columns
  const featureCols = info.columns.filter(c => c !== targetCol && info.columnTypes[c] !== 'identifier');
  if (featureCols.length === 0) {
    issues.push('No usable feature columns found after excluding target and identifiers.');
  }
  
  const missingTargetCount = info.data.filter(row => row[targetCol] === null || row[targetCol] === undefined || row[targetCol] === '').length;
  if (missingTargetCount > 0) {
    warnings.push(`${missingTargetCount} rows have missing target values and will be excluded from training.`);
  }
  
  const isValid = issues.length === 0;
  return { isValid, issues, warnings, taskType };
}

// Clean dataset
export function cleanDataset(data: DataRow[], targetCol: string, columns: string[], columnTypes: Record<string, string>): CleanedDataset {
  const originalData = [...data.map(row => ({ ...row }))];
  
  // Remove duplicates
  const seen = new Set<string>();
  let removedDuplicates = 0;
  let cleanedData: DataRow[] = [];
  
  for (const row of data) {
    const key = JSON.stringify(row);
    if (!seen.has(key)) {
      seen.add(key);
      cleanedData.push({ ...row });
    } else {
      removedDuplicates++;
    }
  }
  
  // Remove rows with missing target
  let removedMissingTarget = 0;
  cleanedData = cleanedData.filter(row => {
    if (row[targetCol] === null || row[targetCol] === undefined || row[targetCol] === '') {
      removedMissingTarget++;
      return false;
    }
    return true;
  });
  
  // Impute missing values in features
  let imputedValues = 0;
  const excludedColumns: string[] = [];
  
  // Identify columns to exclude
  for (const col of columns) {
    if (col === targetCol) continue;
    if (columnTypes[col] === 'identifier') {
      excludedColumns.push(col);
    }
  }
  
  // Impute numerical columns with median
  for (const col of columns) {
    if (excludedColumns.includes(col) || col === targetCol) continue;
    
    const values = cleanedData.map(row => Number(row[col])).filter(v => !isNaN(v) && isFinite(v));
    if (values.length > 0) {
      const sorted = [...values].sort((a, b) => a - b);
      const median = sorted.length % 2 === 0 
        ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2 
        : sorted[Math.floor(sorted.length / 2)];
      
      for (const row of cleanedData) {
        if (row[col] === null || row[col] === undefined || row[col] === '' || (typeof row[col] === 'number' && isNaN(row[col] as number))) {
          row[col] = median;
          imputedValues++;
        }
      }
    }
  }
  
  // Impute categorical columns with mode
  for (const col of columns) {
    if (excludedColumns.includes(col) || col === targetCol) continue;
    
    const values = cleanedData.map(row => row[col]).filter(v => v !== null && v !== undefined && v !== '');
    if (values.length > 0) {
      const freq: Record<string, number> = {};
      for (const v of values) {
        const key = String(v);
        freq[key] = (freq[key] || 0) + 1;
      }
      const mode = Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];
      
      for (const row of cleanedData) {
        if (row[col] === null || row[col] === undefined || row[col] === '') {
          row[col] = mode;
          imputedValues++;
        }
      }
    }
  }
  
  return {
    data: cleanedData,
    originalData,
    removedDuplicates,
    removedMissingTarget,
    imputedValues,
    excludedColumns
  };
}

// Split data into train/test
export function trainTestSplit(data: DataRow[], targetCol: string, testRatio: number = 0.2, seed: number = 42): { train: DataRow[]; test: DataRow[] } {
  // Simple seeded random shuffle
  const shuffled = [...data];
  let s = seed;
  const random = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  
  const testSize = Math.floor(shuffled.length * testRatio);
  return {
    train: shuffled.slice(0, shuffled.length - testSize),
    test: shuffled.slice(shuffled.length - testSize)
  };
}

// Stratified split with index tracking
export function stratifiedSplit<T>(data: T[], targetGetter: (item: T) => string, testRatio: number = 0.2, seed: number = 42): { train: T[]; test: T[] } {
  const groups: Record<string, T[]> = {};
  
  for (const item of data) {
    const key = targetGetter(item);
    if (!groups[key]) groups[key] = [];
    groups[key].push(item);
  }
  
  const train: T[] = [];
  const test: T[] = [];
  
  for (const key of Object.keys(groups)) {
    const group = groups[key];
    let s = seed + key.charCodeAt(0);
    const random = () => {
      s = (s * 1664525 + 1013904223) % 4294967296;
      return s / 4294967296;
    };
    
    // Shuffle
    for (let i = group.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [group[i], group[j]] = [group[j], group[i]];
    }
    
    const testSize = Math.floor(group.length * testRatio);
    test.push(...group.slice(0, testSize));
    train.push(...group.slice(testSize));
  }
  
  return { train, test };
}

// Compute correlation matrix
export function computeCorrelation(data: DataRow[], columns: string[]): { columns: string[]; matrix: number[][] } {
  const numericalCols = columns.filter(col => {
    const values = data.map(row => Number(row[col])).filter(v => !isNaN(v));
    return values.length > data.length * 0.5;
  });
  
  const matrix: number[][] = [];
  
  for (let i = 0; i < numericalCols.length; i++) {
    matrix[i] = [];
    for (let j = 0; j < numericalCols.length; j++) {
      if (i === j) {
        matrix[i][j] = 1;
      } else if (j < i) {
        matrix[i][j] = matrix[j][i];
      } else {
        const x = data.map(row => Number(row[numericalCols[i]])).filter(v => !isNaN(v));
        const y = data.map(row => Number(row[numericalCols[j]])).filter(v => !isNaN(v));
        
        const n = Math.min(x.length, y.length);
        const xSlice = x.slice(0, n);
        const ySlice = y.slice(0, n);
        
        const meanX = xSlice.reduce((a, b) => a + b, 0) / n;
        const meanY = ySlice.reduce((a, b) => a + b, 0) / n;
        
        let num = 0, denX = 0, denY = 0;
        for (let k = 0; k < n; k++) {
          const dx = xSlice[k] - meanX;
          const dy = ySlice[k] - meanY;
          num += dx * dy;
          denX += dx * dx;
          denY += dy * dy;
        }
        
        const denom = Math.sqrt(denX * denY);
        matrix[i][j] = denom === 0 ? 0 : num / denom;
      }
    }
  }
  
  return { columns: numericalCols, matrix };
}

// Generate histogram data
export function generateHistogram(values: number[], bins: number = 20): { bin: string; count: number }[] {
  if (values.length === 0) return [];
  
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / bins;
  
  const histogram: { bin: string; count: number }[] = [];
  
  for (let i = 0; i < bins; i++) {
    const binStart = min + i * binWidth;
    const binEnd = binStart + binWidth;
    const count = values.filter(v => v >= binStart && (i === bins - 1 ? v <= binEnd : v < binEnd)).length;
    histogram.push({
      bin: `${binStart.toFixed(1)}-${binEnd.toFixed(1)}`,
      count
    });
  }
  
  return histogram;
}

// Export data as CSV
export function exportToCSV(data: DataRow[], filename: string): void {
  if (data.length === 0) return;
  
  const columns = Object.keys(data[0]);
  const csvContent = [
    columns.join(','),
    ...data.map(row => columns.map(col => {
      const val = row[col];
      if (val === null || val === undefined) return '';
      const str = String(val);
      return str.includes(',') ? `"${str}"` : str;
    }).join(','))
  ].join('\n');
  
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
