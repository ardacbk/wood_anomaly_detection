import React from 'react';
import { 
  Box, 
  Typography, 
  Grid, 
  Paper, 
  Chip,
  useTheme,
  Card,
  Grow,
  Tooltip,
  alpha
} from '@mui/material';
import ImageSearchIcon from '@mui/icons-material/ImageSearch';
import ReportIcon from '@mui/icons-material/Report';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import LibraryAddCheckIcon from '@mui/icons-material/LibraryAddCheck';

const ResultDisplay = ({ result }) => {
  const theme = useTheme();
  const { score, threshold, is_anomaly, status, heatmap, original, model } = result;
  
  const formatModelName = (name) => {
    switch(name) {
      case 'efficientad': return 'EfficientAD';
      case 'glass': return 'GLASS';
      case 'inpformer': return 'INP-Former';
      default: return name.charAt(0).toUpperCase() + name.slice(1);
    }
  };
  
  const getStatusColor = () => is_anomaly ? theme.palette.error.main : theme.palette.success.main;
  
  const getModelColor = () => {
    switch(model) {
      case 'efficientad': return theme.palette.primary.main;
      case 'glass': return theme.palette.secondary.main;
      case 'inpformer': return theme.palette.success.main;
      default: return theme.palette.info.main;
    }
  };
  
  const getScoreText = () => {
    if (is_anomaly) {
      const ratio = score / threshold;
      if (ratio > 2) return "Very high anomaly score";
      if (ratio > 1.5) return "High anomaly score";
      return "Moderate anomaly score";
    } else {
      const ratio = score / threshold;
      if (ratio < 0.5) return "Very low anomaly score";
      if (ratio < 0.8) return "Low anomaly score";
      return "Score near threshold";
    }
  };

  return (
    <Grow in={true} timeout={800}>
      <Box sx={{ 
        mt: 5,
        width: '100%',
        maxWidth: '1440px',
        mx: 'auto',
        px: { xs: 2, md: 4 }
      }}>
        <Typography 
          variant="h4" 
          gutterBottom 
          sx={{ 
            mb: 4, 
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            color: getStatusColor()
          }}
        >
          <ImageSearchIcon sx={{ mr: 1.5, fontSize: 32 }} />
          Analysis Results
        </Typography>
        
        {/* Analysis Result Summary (Status) */}
        <Paper 
          elevation={3} 
          sx={{ 
            p: 0,
            backgroundColor: alpha(getStatusColor(), 0.03),
            borderRadius: 3,
            border: `1px solid ${alpha(getStatusColor(), 0.2)}`,
            mb: 4
          }}
        >
          <Box sx={{ 
            p: 3,
            display: 'flex',
            alignItems: 'center'
          }}>
            <Box sx={{ 
              backgroundColor: alpha(getStatusColor(), 0.1),
              p: 1.5,
              borderRadius: '50%',
              mr: 2
            }}>
              {is_anomaly ? (
                <ReportIcon sx={{ fontSize: 40, color: getStatusColor() }} />
              ) : (
                <LibraryAddCheckIcon sx={{ fontSize: 40, color: getStatusColor() }} />
              )}
            </Box>
            <Box>
              <Typography variant="h5" component="span" color={getStatusColor()} fontWeight="bold">
                {status}
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mt: 0.5 }}>
                {is_anomaly ? 
                  "Anomalous patterns detected above threshold" :
                  "Image analyzed - no significant anomalies detected"
                }
              </Typography>
            </Box>
            <Box sx={{ ml: 'auto', display: 'flex', gap: 3 }}>
              {[['Model', formatModelName(model), getModelColor()],
                ['Anomaly Score', score.toFixed(2), getStatusColor()],
                ['Threshold', threshold, 'text.primary']].map(([label, value, color], index) => (
                <Box key={index}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {label}
                  </Typography>
                  {label === 'Model' ? (
                    <Chip 
                      label={value}
                      sx={{ 
                        fontWeight: 600,
                        bgcolor: alpha(color, 0.1),
                        color: color,
                        borderRadius: 1.5,
                      }}
                    />
                  ) : (
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Typography 
                        variant="h6" 
                        fontWeight="medium"
                        color={color}
                        sx={{ mr: 1 }}
                      >
                        {value}
                      </Typography>
                      {label === 'Anomaly Score' && (
                        <Tooltip title={getScoreText()} arrow>
                          <InfoOutlinedIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
                        </Tooltip>
                      )}
                    </Box>
                  )}
                </Box>
              ))}
            </Box>
          </Box>
        </Paper>
        
        {/* Images Container - Side by Side and Centered */}
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 4 }}>
          {/* Original Image */}
          <Card sx={{ 
            width: '45%',
            borderRadius: 3,
            overflow: 'hidden'
          }}>
            <Box sx={{ 
              p: 2, 
              bgcolor: 'grey.50',
              borderBottom: `1px solid ${theme.palette.grey[200]}`
            }}>
              <Typography variant="h6" fontWeight={600}>
                Original Image
              </Typography>
            </Box>
            <Box sx={{ 
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              p: 2,
              bgcolor: 'black',
              height: 400
            }}>
              <img 
                src={`data:image/png;base64,${original}`} 
                alt="Original" 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '100%',
                  objectFit: 'contain',
                  borderRadius: 8,
                }}
              />
            </Box>
          </Card>
          
          {/* Heatmap Image */}
          <Card sx={{ 
            width: '45%',
            borderRadius: 3,
            overflow: 'hidden'
          }}>
            <Box sx={{ 
              p: 2, 
              bgcolor: alpha(is_anomaly ? theme.palette.error.main : theme.palette.grey[500], 0.1),
              borderBottom: `1px solid ${theme.palette.grey[200]}`
            }}>
              <Typography 
                variant="h6" 
                fontWeight={600}
                sx={{ color: is_anomaly ? 'error.main' : 'grey.800' }}
              >
                Anomaly Heatmap
                {is_anomaly && (
                  <Chip 
                    size="small" 
                    label="Anomaly Detected" 
                    color="error" 
                    sx={{ ml: 1.5, fontWeight: 500 }}
                  />
                )}
              </Typography>
            </Box>
            <Box sx={{ 
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              p: 2,
              bgcolor: 'black',
              height: 400
            }}>
              <img 
                src={`data:image/png;base64,${heatmap}`} 
                alt="Heatmap" 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '100%',
                  objectFit: 'contain',
                  borderRadius: 8
                }}
              />
            </Box>
          </Card>
        </Box>
      </Box>
    </Grow>
  );
};

export default ResultDisplay;