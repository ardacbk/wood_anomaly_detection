import React from 'react';
import { Slider, Box, Typography, Paper } from '@mui/material';

// Standalone fully functional ThresholdSlider component
const ThresholdSlider = ({ value, onChange, selectedModel }) => {
  const handleChange = (event, newValue) => {
    if (onChange) {
      onChange(newValue);
    }
  };

  // Configure min, max, and step based on selected model
  const getSliderConfig = () => {
    if (selectedModel === 'inpformer') {
      return {
        min: 0,
        max: 1.0,
        step: 0.01,
        marks: [
          { value: 0, label: '0' },
          { value: 0.35, label: '0.35' },
          { value: 1.0, label: '1.0' }
        ]
      };
    }
    else if (selectedModel === 'glass') {
      return {
        min: 0,
        max: 1.0,
        step: 0.01,
        marks: [
          { value: 0, label: '0' },
          { value: 1.0, label: '1.0' }
        ]
      };
    }
     else {
      // Default config for efficientad and other models
      return {
        min: 0.5,
        max: 5.0,
        step: 0.05,
        marks: [
          { value: 0.5, label: '0.5' },
          { value: 2.05, label: '2.05' },
          { value: 5.0, label: '5.0' }
        ]
      };
    }
  };

  const sliderConfig = getSliderConfig();

  return (
    <Paper elevation={1} sx={{ p: 2, height: '100%' }}>
      <Typography variant="body1" gutterBottom>
        Detection Threshold: <strong>{selectedModel === 'inpformer' ? value.toFixed(2) : value.toFixed(2)}</strong>
      </Typography>
      
      <Box sx={{ px: 1, py: 2 }}>
        <Slider
          value={value}
          min={sliderConfig.min}
          max={sliderConfig.max}
          step={sliderConfig.step}
          onChange={handleChange}
          marks={sliderConfig.marks}
          valueLabelDisplay="auto"
        />
      </Box>
    </Paper>
  );
};

export default ThresholdSlider;