import React from 'react';
import { 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Box,
  Paper,
  Typography,
  Chip,
  useTheme
} from '@mui/material';
import AssessmentIcon from '@mui/icons-material/Assessment';
import MemoryIcon from '@mui/icons-material/Memory';
import ScienceIcon from '@mui/icons-material/Science';

const ModelSelector = ({ models, selectedModel, onChange }) => {
  const theme = useTheme();
  
  const formatModelName = (model) => {
    switch(model) {
      case 'efficientad':
        return 'EfficientAD';
      case 'glass':
        return 'GLASS';
      case 'inpformer':
        return 'INP-Former';
      default:
        return model.charAt(0).toUpperCase() + model.slice(1);
    }
  };
  
  const getModelDescription = (model) => {
    switch(model) {
      case 'efficientad':
        return 'Fast and efficient anomaly detection';
      case 'glass':
        return 'High precision global analysis';
      case 'inpformer':
        return 'Advanced transformer-based detection';
      default:
        return 'AI-powered anomaly detection';
    }
  };
  
  const getModelIcon = (model) => {
    switch(model) {
      case 'efficientad':
        return <MemoryIcon sx={{ color: theme.palette.primary.main }} />;
      case 'glass':
        return <ScienceIcon sx={{ color: theme.palette.secondary.main }} />;
      case 'inpformer':
        return <AssessmentIcon sx={{ color: theme.palette.success.main }} />;
      default:
        return <MemoryIcon />;
    }
  };

  return (
    <Paper 
      elevation={1} 
      sx={{ 
        p: 2,
        height: '100%',
        borderRadius: 2,
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <Typography 
        variant="body1" 
        gutterBottom
        sx={{ fontWeight: 500, mb: 2 }}
      >
        Analysis Model
      </Typography>
      
      <FormControl fullWidth variant="outlined">
        <InputLabel id="model-select-label">Select Model</InputLabel>
        <Select
          labelId="model-select-label"
          id="model-select"
          value={selectedModel}
          label="Select Model"
          onChange={onChange}
          sx={{ 
            borderRadius: 1.5,
            '& .MuiOutlinedInput-notchedOutline': {
              borderColor: theme.palette.grey[300]
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              borderColor: theme.palette.primary.light
            }
          }}
        >
          {models.map((model) => (
            <MenuItem key={model} value={model}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Box sx={{ mr: 1.5 }}>
                  {getModelIcon(model)}
                </Box>
                <Box>
                  <Typography variant="body1">
                    {formatModelName(model)}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    {getModelDescription(model)}
                  </Typography>
                </Box>
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      
      {selectedModel && (
        <Box sx={{ mt: 2, display: 'flex', alignItems: 'center' }}>
          <Chip 
            icon={getModelIcon(selectedModel)}
            label={`Using: ${formatModelName(selectedModel)}`}
            variant="outlined"
            sx={{ 
              borderRadius: 1.5,
              backgroundColor: 'rgba(25, 118, 210, 0.08)',
              fontWeight: 500
            }}
          />
        </Box>
      )}
    </Paper>
  );
};

export default ModelSelector;