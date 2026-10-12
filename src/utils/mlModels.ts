import { DataRow } from './dataUtils';

// Types
export interface ModelResult {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  specificity: number;
  aucRoc: number;
  confusionMatrix: number[][];
  predictions: number[];
  probabilities: number[];
  trueLabels: number[];
  featureImportance: Record<string, number>;
  trainingTime: number;
}

export interface TrainedModel {
  name: string;
  predict: (features: number[]) => number;
  predictProba: (features: number[]) => number;
  featureImportance: Record<string, number>;
  coefficients?: number[];
  classes: number[];
}

// Utility functions
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
}

function dotProduct(a: number[], b: number[]): number {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

// Logistic Regression
export function trainLogisticRegression(X: number[][], y: number[], learningRate: number = 0.1, epochs: number = 1000, regularization: number = 0.01): TrainedModel {
  const nFeatures = X[0].length;
  let weights = new Array(nFeatures).fill(0);
  let bias = 0;
  
  const startTime = performance.now();
  
  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = 0; i < X.length; i++) {
      const z = dotProduct(X[i], weights) + bias;
      const prediction = sigmoid(z);
      const error = y[i] - prediction;
      
      for (let j = 0; j < nFeatures; j++) {
        weights[j] += learningRate * (error * X[i][j] - regularization * weights[j]);
      }
      bias += learningRate * error;
    }
  }
  
  const trainingTime = performance.now() - startTime;
  
  // Feature importance based on absolute weight values
  const maxWeight = Math.max(...weights.map(Math.abs), 0.001);
  const featureImportance: Record<string, number> = {};
  weights.forEach((w, i) => {
    featureImportance[`feature_${i}`] = Math.abs(w) / maxWeight;
  });
  
  return {
    name: 'Logistic Regression',
    predict: (features: number[]) => sigmoid(dotProduct(features, weights) + bias) >= 0.5 ? 1 : 0,
    predictProba: (features: number[]) => sigmoid(dotProduct(features, weights) + bias),
    featureImportance,
    coefficients: weights,
    classes: [0, 1]
  };
}

// Decision Tree Node
interface TreeNode {
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  value?: number;
  isLeaf: boolean;
}

function giniImpurity(y: number[]): number {
  if (y.length === 0) return 0;
  const counts: Record<number, number> = {};
  for (const v of y) counts[v] = (counts[v] || 0) + 1;
  let impurity = 1;
  for (const count of Object.values(counts)) {
    const p = count / y.length;
    impurity -= p * p;
  }
  return impurity;
}

function findBestSplit(X: number[][], y: number[], featureIndices: number[]): { featureIndex: number; threshold: number; gain: number } | null {
  const parentGini = giniImpurity(y);
  let bestGain = 0;
  let bestFeature = -1;
  let bestThreshold = 0;
  
  for (const fi of featureIndices) {
    const values = X.map(row => row[fi]);
    const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
    
    for (let i = 0; i < uniqueValues.length - 1; i++) {
      const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
      
      const leftY: number[] = [];
      const rightY: number[] = [];
      
      for (let j = 0; j < X.length; j++) {
        if (X[j][fi] <= threshold) {
          leftY.push(y[j]);
        } else {
          rightY.push(y[j]);
        }
      }
      
      if (leftY.length === 0 || rightY.length === 0) continue;
      
      const leftGini = giniImpurity(leftY);
      const rightGini = giniImpurity(rightY);
      const weightedGini = (leftY.length * leftGini + rightY.length * rightGini) / y.length;
      const gain = parentGini - weightedGini;
      
      if (gain > bestGain) {
        bestGain = gain;
        bestFeature = fi;
        bestThreshold = threshold;
      }
    }
  }
  
  if (bestFeature === -1) return null;
  return { featureIndex: bestFeature, threshold: bestThreshold, gain: bestGain };
}

function buildTree(X: number[][], y: number[], depth: number, maxDepth: number, minSamples: number, featureIndices: number[]): TreeNode {
  const counts: Record<number, number> = {};
  for (const v of y) counts[v] = (counts[v] || 0) + 1;
  
  if (depth >= maxDepth || y.length < minSamples || Object.keys(counts).length === 1) {
    const majority = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    return { isLeaf: true, value: Number(majority) };
  }
  
  const split = findBestSplit(X, y, featureIndices);
  if (!split) {
    const majority = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    return { isLeaf: true, value: Number(majority) };
  }
  
  const leftX: number[][] = [];
  const leftY: number[] = [];
  const rightX: number[][] = [];
  const rightY: number[] = [];
  
  for (let i = 0; i < X.length; i++) {
    if (X[i][split.featureIndex] <= split.threshold) {
      leftX.push(X[i]);
      leftY.push(y[i]);
    } else {
      rightX.push(X[i]);
      rightY.push(y[i]);
    }
  }
  
  return {
    isLeaf: false,
    featureIndex: split.featureIndex,
    threshold: split.threshold,
    left: buildTree(leftX, leftY, depth + 1, maxDepth, minSamples, featureIndices),
    right: buildTree(rightX, rightY, depth + 1, maxDepth, minSamples, featureIndices)
  };
}

function predictTree(node: TreeNode, features: number[]): number {
  if (node.isLeaf) return node.value!;
  if (features[node.featureIndex!] <= node.threshold!) {
    return predictTree(node.left!, features);
  }
  return predictTree(node.right!, features);
}

// Decision Tree Classifier
export function trainDecisionTree(X: number[][], y: number[], maxDepth: number = 10, minSamples: number = 5): TrainedModel {
  const startTime = performance.now();
  const nFeatures = X[0].length;
  const featureIndices = Array.from({ length: nFeatures }, (_, i) => i);
  
  const tree = buildTree(X, y, 0, maxDepth, minSamples, featureIndices);
  const trainingTime = performance.now() - startTime;
  
  // Feature importance based on how often features are used for splitting
  const featureCounts: Record<number, number> = {};
  function countFeatures(node: TreeNode) {
    if (node.isLeaf) return;
    featureCounts[node.featureIndex!] = (featureCounts[node.featureIndex!] || 0) + 1;
    countFeatures(node.left!);
    countFeatures(node.right!);
  }
  countFeatures(tree);
  
  const maxCount = Math.max(...Object.values(featureCounts), 1);
  const featureImportance: Record<string, number> = {};
  for (let i = 0; i < nFeatures; i++) {
    featureImportance[`feature_${i}`] = (featureCounts[i] || 0) / maxCount;
  }
  
  return {
    name: 'Decision Tree',
    predict: (features: number[]) => predictTree(tree, features),
    predictProba: (features: number[]) => predictTree(tree, features),
    featureImportance,
    classes: [0, 1]
  };
}

// Random Forest Classifier
export function trainRandomForest(X: number[][], y: number[], nTrees: number = 50, maxDepth: number = 8, minSamples: number = 5, seed: number = 42): TrainedModel {
  const startTime = performance.now();
  const trees: TreeNode[] = [];
  const nFeatures = X[0].length;
  const sampleSize = Math.floor(X.length * 0.8);
  
  let s = seed;
  const random = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  
  for (let t = 0; t < nTrees; t++) {
    // Bootstrap sample
    const indices: number[] = [];
    for (let i = 0; i < sampleSize; i++) {
      indices.push(Math.floor(random() * X.length));
    }
    
    const bootstrapX = indices.map(i => X[i]);
    const bootstrapY = indices.map(i => y[i]);
    
    // Random feature subset
    const nFeatureSubset = Math.max(1, Math.floor(Math.sqrt(nFeatures)));
    const featureIndices: number[] = [];
    const allFeatures = Array.from({ length: nFeatures }, (_, i) => i);
    for (let i = 0; i < nFeatureSubset; i++) {
      const idx = Math.floor(random() * allFeatures.length);
      featureIndices.push(allFeatures[idx]);
      allFeatures.splice(idx, 1);
    }
    
    const tree = buildTree(bootstrapX, bootstrapY, 0, maxDepth, minSamples, featureIndices);
    trees.push(tree);
  }
  
  const trainingTime = performance.now() - startTime;
  
  // Aggregate feature importance
  const featureImportance: Record<string, number> = {};
  for (let i = 0; i < nFeatures; i++) {
    featureImportance[`feature_${i}`] = 0;
  }
  
  for (const tree of trees) {
    function countFeatures(node: TreeNode) {
      if (node.isLeaf) return;
      featureImportance[`feature_${node.featureIndex!}`] += 1;
      countFeatures(node.left!);
      countFeatures(node.right!);
    }
    countFeatures(tree);
  }
  
  const maxImportance = Math.max(...Object.values(featureImportance), 1);
  for (const key of Object.keys(featureImportance)) {
    featureImportance[key] /= maxImportance;
  }
  
  return {
    name: 'Random Forest',
    predict: (features: number[]) => {
      const votes: Record<number, number> = {};
      for (const tree of trees) {
        const pred = predictTree(tree, features);
        votes[pred] = (votes[pred] || 0) + 1;
      }
      return Number(Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0]);
    },
    predictProba: (features: number[]) => {
      const votes: Record<number, number> = {};
      for (const tree of trees) {
        const pred = predictTree(tree, features);
        votes[pred] = (votes[pred] || 0) + 1;
      }
      return (votes[1] || 0) / nTrees;
    },
    featureImportance,
    classes: [0, 1]
  };
}

// Gradient Boosting Classifier
export function trainGradientBoosting(X: number[][], y: number[], nTrees: number = 100, learningRate: number = 0.1, maxDepth: number = 4): TrainedModel {
  const startTime = performance.now();
  const nFeatures = X[0].length;
  
  // Initialize with log-odds
  const posCount = y.filter(v => v === 1).length;
  const negCount = y.length - posCount;
  let initialPrediction = Math.log(posCount / negCount);
  
  // Store tree predictions
  const trees: { tree: TreeNode; weight: number }[] = [];
  let currentPredictions = new Array(X.length).fill(initialPrediction);
  
  for (let t = 0; t < nTrees; t++) {
    // Compute pseudo-residuals
    const residuals = y.map((yi, i) => yi - sigmoid(currentPredictions[i]));
    
    // Build tree on residuals
    const tree = buildTree(X, residuals, 0, maxDepth, 5, Array.from({ length: nFeatures }, (_, i) => i));
    
    // Compute leaf values (optimal step size)
    const treePredictions = X.map(row => predictTree(tree, row));
    
    // Simple: use learning rate as weight
    trees.push({ tree, weight: learningRate });
    
    // Update predictions
    for (let i = 0; i < X.length; i++) {
      currentPredictions[i] += learningRate * treePredictions[i];
    }
  }
  
  const trainingTime = performance.now() - startTime;
  
  // Feature importance
  const featureImportance: Record<string, number> = {};
  for (let i = 0; i < nFeatures; i++) {
    featureImportance[`feature_${i}`] = 0;
  }
  
  for (const { tree } of trees) {
    function countFeatures(node: TreeNode) {
      if (node.isLeaf) return;
      featureImportance[`feature_${node.featureIndex!}`] += 1;
      countFeatures(node.left!);
      countFeatures(node.right!);
    }
    countFeatures(tree);
  }
  
  const maxImportance = Math.max(...Object.values(featureImportance), 1);
  for (const key of Object.keys(featureImportance)) {
    featureImportance[key] /= maxImportance;
  }
  
  const predictFn = (features: number[]): number => {
    let prediction = initialPrediction;
    for (const { tree, weight } of trees) {
      prediction += weight * predictTree(tree, features);
    }
    return sigmoid(prediction) >= 0.5 ? 1 : 0;
  };
  
  const predictProbaFn = (features: number[]): number => {
    let prediction = initialPrediction;
    for (const { tree, weight } of trees) {
      prediction += weight * predictTree(tree, features);
    }
    return sigmoid(prediction);
  };
  
  return {
    name: 'Gradient Boosting',
    predict: predictFn,
    predictProba: predictProbaFn,
    featureImportance,
    classes: [0, 1]
  };
}

// KNN Classifier
export function trainKNN(X: number[][], y: number[], k: number = 5): TrainedModel {
  const startTime = performance.now();
  const trainingTime = performance.now() - startTime;
  
  // Normalize features for KNN
  const means = X[0].map((_, j) => X.reduce((sum, row) => sum + row[j], 0) / X.length);
  const stds = X[0].map((_, j) => {
    const mean = means[j];
    return Math.sqrt(X.reduce((sum, row) => sum + Math.pow(row[j] - mean, 2), 0) / X.length) || 1;
  });
  
  const normalizedX = X.map(row => row.map((v, j) => (v - means[j]) / stds[j]));
  
  return {
    name: 'K-Nearest Neighbors',
    predict: (features: number[]) => {
      const normalized = features.map((v, j) => (v - means[j]) / stds[j]);
      const distances = normalizedX.map((row, i) => ({
        distance: Math.sqrt(row.reduce((sum, v, j) => sum + Math.pow(v - normalized[j], 2), 0)),
        label: y[i]
      }));
      distances.sort((a, b) => a.distance - b.distance);
      const kNearest = distances.slice(0, k);
      const votes: Record<number, number> = {};
      for (const { label } of kNearest) {
        votes[label] = (votes[label] || 0) + 1;
      }
      return Number(Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0]);
    },
    predictProba: (features: number[]) => {
      const normalized = features.map((v, j) => (v - means[j]) / stds[j]);
      const distances = normalizedX.map((row, i) => ({
        distance: Math.sqrt(row.reduce((sum, v, j) => sum + Math.pow(v - normalized[j], 2), 0)),
        label: y[i]
      }));
      distances.sort((a, b) => a.distance - b.distance);
      const kNearest = distances.slice(0, k);
      const posCount = kNearest.filter(d => d.label === 1).length;
      return posCount / k;
    },
    featureImportance: {},
    classes: [0, 1]
  };
}

// Evaluate model
export function evaluateModel(model: TrainedModel, X: number[][], y: number[]): ModelResult {
  const predictions: number[] = [];
  const probabilities: number[] = [];
  
  for (const features of X) {
    predictions.push(model.predict(features));
    probabilities.push(model.predictProba(features));
  }
  
  // Confusion matrix
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < y.length; i++) {
    if (y[i] === 1 && predictions[i] === 1) tp++;
    else if (y[i] === 0 && predictions[i] === 0) tn++;
    else if (y[i] === 0 && predictions[i] === 1) fp++;
    else fn++;
  }
  
  const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
  const specificity = tn / (tn + fp) || 0;
  
  // AUC-ROC calculation
  const aucRoc = computeAUC(y, probabilities);
  
  return {
    name: model.name,
    accuracy,
    precision,
    recall,
    f1Score,
    specificity,
    aucRoc,
    confusionMatrix: [[tn, fp], [fn, tp]],
    predictions,
    probabilities,
    trueLabels: y,
    featureImportance: model.featureImportance,
    trainingTime: 0
  };
}

// Compute AUC-ROC
function computeAUC(y: number[], scores: number[]): number {
  const pairs: { score: number; label: number }[] = y.map((label, i) => ({ score: scores[i], label }));
  pairs.sort((a, b) => b.score - a.score);
  
  let tp = 0, fp = 0;
  const totalPos = y.filter(v => v === 1).length;
  const totalNeg = y.filter(v => v === 0).length;
  
  if (totalPos === 0 || totalNeg === 0) return 0.5;
  
  let auc = 0;
  let prevFPR = 0;
  let prevTPR = 0;
  
  for (const pair of pairs) {
    if (pair.label === 1) tp++;
    else fp++;
    
    const tpr = tp / totalPos;
    const fpr = fp / totalNeg;
    
    auc += (fpr - prevFPR) * (tpr + prevTPR) / 2;
    prevFPR = fpr;
    prevTPR = tpr;
  }
  
  return auc;
}

// Compute ROC curve points
export function computeROCCurve(y: number[], scores: number[]): { fpr: number; tpr: number; threshold: number }[] {
  const pairs: { score: number; label: number }[] = y.map((label, i) => ({ score: scores[i], label }));
  pairs.sort((a, b) => b.score - a.score);
  
  const totalPos = y.filter(v => v === 1).length;
  const totalNeg = y.filter(v => v === 0).length;
  
  if (totalPos === 0 || totalNeg === 0) return [{ fpr: 0, tpr: 0, threshold: 0 }];
  
  const points: { fpr: number; tpr: number; threshold: number }[] = [{ fpr: 0, tpr: 0, threshold: 1 }];
  let tp = 0, fp = 0;
  
  for (const pair of pairs) {
    if (pair.label === 1) tp++;
    else fp++;
    points.push({
      fpr: fp / totalNeg,
      tpr: tp / totalPos,
      threshold: pair.score
    });
  }
  
  return points;
}

// Prepare features from data
export function prepareFeatures(data: DataRow[], featureCols: string[], columnTypes: Record<string, string>): { X: number[][]; featureNames: string[]; encoders: Record<string, Record<string, number>> } {
  const X: number[][] = [];
  const featureNames: string[] = [];
  const encoders: Record<string, Record<string, number>> = {};
  
  for (const col of featureCols) {
    if (columnTypes[col] === 'numerical' || columnTypes[col] === 'binary') {
      featureNames.push(col);
    } else if (columnTypes[col] === 'categorical') {
      // One-hot encode
      const uniqueVals = [...new Set(data.map(row => String(row[col])).filter(v => v !== '' && v !== 'null'))];
      const encoder: Record<string, number> = {};
      uniqueVals.forEach((v, i) => { encoder[v] = i; });
      encoders[col] = encoder;
      for (const v of uniqueVals) {
        featureNames.push(`${col}_${v}`);
      }
    }
  }
  
  for (const row of data) {
    const features: number[] = [];
    for (const col of featureCols) {
      if (columnTypes[col] === 'numerical' || columnTypes[col] === 'binary') {
        features.push(Number(row[col]) || 0);
      } else if (columnTypes[col] === 'categorical' && encoders[col]) {
        const val = String(row[col]);
        const uniqueVals = Object.keys(encoders[col]);
        for (const v of uniqueVals) {
          features.push(val === v ? 1 : 0);
        }
      }
    }
    X.push(features);
  }
  
  return { X, featureNames, encoders };
}

// Scale features
export function scaleFeatures(X: number[][]): { scaled: number[][]; means: number[]; stds: number[] } {
  const nFeatures = X[0].length;
  const means: number[] = [];
  const stds: number[] = [];
  
  for (let j = 0; j < nFeatures; j++) {
    const values = X.map(row => row[j]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length) || 1;
    means.push(mean);
    stds.push(std);
  }
  
  const scaled = X.map(row => row.map((v, j) => (v - means[j]) / stds[j]));
  return { scaled, means, stds };
}
