import React from 'react';
import { FormControl, InputLabel, Select, MenuItem } from '@mui/material';

const ModelSelector = ({ models, selectedModel, onChange }) => {
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

  return (
    <FormControl fullWidth>
      <InputLabel id="model-select-label">Select Model</InputLabel>
      <Select
        labelId="model-select-label"
        value={selectedModel}
        label="Select Model"
        onChange={onChange}
      >
        {models.map((model) => (
          <MenuItem key={model} value={model}>
            {formatModelName(model)}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
};

export default ModelSelector;