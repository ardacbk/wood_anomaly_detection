import React from 'react';
import { Box, Typography, Grid, Paper, Divider } from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

const ResultDisplay = ({ result }) => {
  const { score, threshold, is_anomaly, status, heatmap, original, model } = result;
  
  // Format model name for display
  const formatModelName = (name) => {
    switch(name) {
      case 'efficientad':
        return 'EfficientAD';
      case 'glass':
        return 'GLASS';
      case 'inpformer':
        return 'INP-Former';
      default:
        return name.charAt(0).toUpperCase() + name.slice(1);
    }
  };
  
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        Analysis Results
      </Typography>
      
      <Grid container spacing={3}>
        {/* Images Section */}
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={2} 
            sx={{ 
              p: 2, 
              height: '100%', 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center'
            }}
          >
            <Typography variant="h6" gutterBottom>Original Image</Typography>
            <Box sx={{ mt: 2, mb: 2, textAlign: 'center' }}>
              <img 
                src={`data:image/png;base64,${original}`} 
                alt="Original" 
                style={{ maxWidth: '100%', maxHeight: '300px' }}
              />
            </Box>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={2} 
            sx={{ 
              p: 2, 
              height: '100%', 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center'
            }}
          >
            <Typography variant="h6" gutterBottom>Anomaly Heatmap</Typography>
            <Box sx={{ mt: 2, mb: 2, textAlign: 'center' }}>
              <img 
                src={`data:image/png;base64,${heatmap}`} 
                alt="Heatmap" 
                style={{ maxWidth: '100%', maxHeight: '300px' }}
              />
            </Box>
          </Paper>
        </Grid>
        
        {/* Results Section */}
        <Grid item xs={12}>
          <Paper 
            elevation={3} 
            sx={{ 
              p: 3, 
              backgroundColor: is_anomaly ? '#ffebee' : '#e8f5e9',
              border: `2px solid ${is_anomaly ? '#d32f2f' : '#2e7d32'}`,
              borderRadius: 2
            }}
          >
            <Grid container alignItems="center" spacing={2}>
              <Grid item>
                {is_anomaly ? (
                  <ErrorOutlineIcon color="error" fontSize="large" />
                ) : (
                  <CheckCircleOutlineIcon color="success" fontSize="large" />
                )}
              </Grid>
              <Grid item>
                <Typography 
                  variant="h5" 
                  component="span" 
                  color={is_anomaly ? 'error' : 'success'}
                  fontWeight="bold"
                >
                  {status}
                </Typography>
              </Grid>
            </Grid>
            
            <Divider sx={{ my: 2 }} />
            
            <Grid container spacing={3}>
              <Grid item xs={12} sm={4}>
                <Typography variant="body1" gutterBottom>
                  <strong>Model:</strong> {formatModelName(model)}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="body1" gutterBottom>
                  <strong>Anomaly Score:</strong> {score.toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="body1" gutterBottom>
                  <strong>Threshold:</strong> {threshold}
                </Typography>
              </Grid>
            </Grid>
            
            <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
              {is_anomaly ? 
                "The image has been analyzed and anomalous patterns were detected above the threshold. Review the heatmap to see affected regions." :
                "The image has been analyzed and no significant anomalies were detected. The image appears to be normal."
              }
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ResultDisplay;