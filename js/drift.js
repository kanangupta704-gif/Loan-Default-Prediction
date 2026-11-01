/**
 * Drift & Data Distribution Monitoring (EvidentlyAI / MLflow Integration)
 * Tracks Population Stability Index (PSI) and Kolmogorov-Smirnov feature drift.
 */

const DriftEngine = {
  // Production monitoring snapshot (last 30 days vs training baseline)
  monitorSummary: {
    overallPsi: 0.078,
    status: 'HEALTHY',
    statusLabel: 'No Significant Drift Detected (PSI < 0.10)',
    driftAlertCount: 0,
    monitoredInferences: 34820,
    lastScanTime: '24 minutes ago',
    pipelineVersion: 'EvidentlyAI 0.4.12 / MLflow 2.9'
  },

  // Per-feature distribution drift statistics
  featureMetrics: [
    {
      feature: 'Debt-to-Income (DTI)',
      type: 'Numerical',
      psi: 0.089,
      ksStat: 0.042,
      pValue: 0.38,
      status: 'STABLE',
      shiftDirection: '+2.1% median shift due to macro inflation'
    },
    {
      feature: 'Credit Score (FICO)',
      type: 'Numerical',
      psi: 0.042,
      ksStat: 0.021,
      pValue: 0.65,
      status: 'STABLE',
      shiftDirection: 'Minimal change in applicant credit quality'
    },
    {
      feature: 'Credit Inquiries (1 yr)',
      type: 'Numerical',
      psi: 0.118,
      ksStat: 0.068,
      pValue: 0.08,
      status: 'MODERATE',
      shiftDirection: 'Moderate uptick in external loan shopping (+14%)'
    },
    {
      feature: 'Revolving Line Utilization',
      type: 'Numerical',
      psi: 0.065,
      ksStat: 0.034,
      pValue: 0.47,
      status: 'STABLE',
      shiftDirection: 'Consistent with seasonal credit usage'
    },
    {
      feature: 'Monthly Gross Income',
      type: 'Numerical',
      psi: 0.051,
      ksStat: 0.028,
      pValue: 0.52,
      status: 'STABLE',
      shiftDirection: 'Normal nominal wage growth adjustment'
    },
    {
      feature: 'Loan Purpose',
      type: 'Categorical',
      psi: 0.038,
      ksStat: 0.019,
      pValue: 0.81,
      status: 'STABLE',
      shiftDirection: 'Debt consolidation remains top category (58%)'
    }
  ],

  // Binned distribution data for DTI (Baseline training vs Production last 30d)
  dtiDistributionBins: [
    { bin: '0 - 10%', baseline: 8.5, production: 7.2 },
    { bin: '10 - 20%', baseline: 26.2, production: 24.8 },
    { bin: '20 - 30%', baseline: 34.0, production: 33.5 },
    { bin: '30 - 40%', baseline: 21.5, production: 23.4 },
    { bin: '40 - 50%', baseline: 7.8, production: 8.9 },
    { bin: '> 50%', baseline: 2.0, production: 2.2 }
  ],

  /**
   * Simulates an intentional macro shock to demonstrate automated alert triggers
   */
  simulateMacroShift(applyShock = false) {
    if (applyShock) {
      return {
        overallPsi: 0.284,
        status: 'DRIFT_ALERT',
        statusLabel: 'CRITICAL: Severe Population Shift (PSI >= 0.25). Model retraining initiated.',
        alertColor: '#DC2626',
        badgeBg: '#FEE2E2',
        retrainTriggered: true
      };
    } else {
      return {
        overallPsi: 0.078,
        status: 'HEALTHY',
        statusLabel: 'STABLE: Population distributions aligned with training baseline (PSI < 0.10).',
        alertColor: '#15803D',
        badgeBg: '#DCFCE7',
        retrainTriggered: false
      };
    }
  }
};

