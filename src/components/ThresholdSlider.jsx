import React from 'react';
import { Slider, Box, Typography, Paper, useTheme } from '@mui/material';
import TuneIcon from '@mui/icons-material/Tune';

const ThresholdSlider = ({ value, onChange, selectedModel }) => {
  const theme = useTheme();

  const getSliderConfig = () => {
    if (selectedModel === 'inpformer') {
      return { min: 0, max: 1.0, step: 0.01, color: theme.palette.success.main };
    }
    if (selectedModel === 'glass') {
      return { min: 0, max: 1.0, step: 0.01, color: theme.palette.secondary.main };
    }
    return { min: 0.5, max: 5.0, step: 0.05, color: theme.palette.primary.main };
  };

  const { min, max, step, color } = getSliderConfig();

  return (
    <Paper sx={{ 
      p: 3, 
      borderRadius: 3,
      width: '100%',
      maxWidth: 600,
      mx: 'auto',
      boxShadow: theme.shadows[3]
    }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <TuneIcon sx={{ color, fontSize: 28 }} />
        <Typography variant="h6" fontWeight={600}>
          Detection Sensitivity
        </Typography>
      </Box>

      <Box sx={{ px: 2, pb: 1, position: 'relative' }}>
        <Slider
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e, val) => onChange(val)}
          valueLabelDisplay="auto"
          valueLabelFormat={(x) => x.toFixed(2)}
          sx={{
            '& .MuiSlider-thumb': {
              width: 24,
              height: 24,
              backgroundColor: '#fff',
              border: `3px solid ${color}`,
              '&:hover, &.Mui-active': { boxShadow: 'none' }
            },
            '& .MuiSlider-valueLabel': {
              backgroundColor: color,
              fontWeight: 700,
              top: -10
            },
            '& .MuiSlider-track': { height: 6, backgroundColor: color },
            '& .MuiSlider-rail': { height: 6, backgroundColor: theme.palette.grey[300] }
          }}
        />
        
        {/* Current Value Display */}
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          mt: 3,
          px: 0.5
        }}>
          <Typography variant="body2" color="textSecondary">
            {min.toFixed(2)}
          </Typography>
          <Typography variant="body2" fontWeight={600} color={color}>
            {value.toFixed(2)}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            {max.toFixed(2)}
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default ThresholdSlider;