import React, { useState, useEffect } from 'react';
import { Container, Box, Typography, Paper, Button, Grid, CircularProgress } from '@mui/material';
import ImageUploader from './components/ImageUploader';
import ResultDisplay from './components/ResultDisplay';
import ModelSelector from './components/ModelSelector';
import ThresholdSlider from './components/ThresholdSlider';

function App() {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [threshold, setThreshold] = useState(2.1); // Default for efficientad
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('efficientad');
  const [previewUrl, setPreviewUrl] = useState(null);

  // Fetch available models when component mounts
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/models');
        const data = await response.json();
        setAvailableModels(data.available_models || []);
        
        // If the default model is available, select it
        if (data.default_model && data.available_models.includes(data.default_model)) {
          setSelectedModel(data.default_model);
          // Set appropriate threshold based on model
          if(data.default_model === 'efficientad'){setThreshold(2.05)}
          if(data.default_model === 'inpformer'){setThreshold(0.35)}
          if(data.default_model === 'glass'){setThreshold(0.9)}
        } else if (data.available_models.length > 0) {
          // Otherwise select the first available model
          const firstModel = data.available_models[0];
          setSelectedModel(firstModel);
          if(data.default_model === 'efficientad'){setThreshold(2.05)}
          if(data.default_model === 'inpformer'){setThreshold(0.35)}
          if(data.default_model === 'glass'){setThreshold(0.9)}
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };

    fetchModels();
  }, []);

  const handleFileChange = (newFile) => {
    setFile(newFile);
    setResult(null);
    
    // Create preview URL for the file
    if (newFile) {
      const reader = new FileReader();
      reader.onload = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(newFile);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleThresholdChange = (newValue) => {
    setThreshold(newValue);
  };

  const handleModelChange = (event) => {
    const newModel = event.target.value;
    setSelectedModel(newModel);
    
    // Update threshold based on selected model
    if (newModel === 'inpformer') {
      setThreshold(0.35);
    } else if (newModel === 'efficientad') {
      setThreshold(2.05);
    }
    else if(newModel === 'glass'){
      setThreshold(0.9)
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setIsLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold);
    formData.append('model', selectedModel);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Error analyzing image: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        <Typography variant="h4" component="h1" align="center" gutterBottom>
          Advanced Anomaly Detection
        </Typography>
        
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Settings
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <ModelSelector 
                models={availableModels} 
                selectedModel={selectedModel} 
                onChange={handleModelChange} 
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <ThresholdSlider 
                value={threshold} 
                onChange={handleThresholdChange}
                selectedModel={selectedModel}
              />
            </Grid>
          </Grid>
        </Box>

        <ImageUploader onFileChange={handleFileChange} />

        {previewUrl && !result && (
          <Box sx={{ mt: 3, textAlign: 'center' }}>
            <img 
              src={previewUrl} 
              alt="Preview" 
              style={{ maxWidth: '100%', maxHeight: '300px' }} 
            />
            
            <Box sx={{ mt: 2 }}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleSubmit}
                disabled={isLoading}
                sx={{ mr: 2 }}
              >
                {isLoading ? <CircularProgress size={24} /> : 'Analyze Image'}
              </Button>
              
              <Button 
                variant="outlined" 
                color="secondary" 
                onClick={handleReset}
                disabled={isLoading}
              >
                Reset
              </Button>
            </Box>
          </Box>
        )}

        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
            <CircularProgress />
            <Typography variant="h6" sx={{ ml: 2 }}>
              Analyzing image...
            </Typography>
          </Box>
        )}

        {result && <ResultDisplay result={result} />}
      </Paper>
    </Container>
  );
}

export default App;