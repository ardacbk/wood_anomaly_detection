import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Box, 
  Typography, 
  Paper, 
  Button, 
  Grid, 
  CircularProgress,
  useTheme,
  Card,
  CardContent,
  Fade,
  Grow
} from '@mui/material';
import ImageUploader from './components/ImageUploader';
import ResultDisplay from './components/ResultDisplay';
import ModelSelector from './components/ModelSelector';
import ThresholdSlider from './components/ThresholdSlider';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import RestartAltIcon from '@mui/icons-material/RestartAlt';

function App() {
  const theme = useTheme();
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
          if(data.default_model === 'glass'){setThreshold(0.91)}
        } else if (data.available_models.length > 0) {
          // Otherwise select the first available model
          const firstModel = data.available_models[0];
          setSelectedModel(firstModel);
          if(firstModel === 'efficientad'){setThreshold(2.05)}
          if(firstModel === 'inpformer'){setThreshold(0.35)}
          if(firstModel === 'glass'){setThreshold(0.91)}
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
      setThreshold(0.91)
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
      <Paper 
        elevation={3} 
        sx={{ 
          p: { xs: 2, md: 4 }, 
          borderRadius: 3,
          background: `linear-gradient(145deg, ${theme.palette.background.paper} 0%, ${theme.palette.background.default} 100%)`,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)'
        }}
      >
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Typography 
            variant="h3" 
            component="h1" 
            gutterBottom
            sx={{
              fontWeight: 700,
              background: 'linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0px 2px 5px rgba(0,0,0,0.1)'
            }}
          >
            Advanced Anomaly Detection
          </Typography>
          <Typography 
            variant="subtitle1" 
            color="textSecondary" 
            sx={{ maxWidth: '600px', mx: 'auto', mb: 3 }}
          >
            Upload an image to analyze and detect potential anomalies using state-of-the-art AI models
          </Typography>
        </Box>
        
        <Card 
          elevation={2} 
          sx={{ 
            mb: 4, 
            borderRadius: 2,
            background: theme.palette.background.paper,
            transition: 'all 0.3s ease'
          }}
        >
          <CardContent sx={{ p: 3 }}>
            <Typography 
              variant="h5" 
              gutterBottom 
              sx={{ 
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                mb: 3
              }}
            >
              <AutoFixHighIcon sx={{ mr: 1 }} />
              Configuration Settings
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
          </CardContent>
        </Card>

        <Box sx={{ mb: 4 }}>
          <ImageUploader onFileChange={handleFileChange} />
        </Box>

        {previewUrl && !result && (
          <Grow in={Boolean(previewUrl)} timeout={500}>
            <Box 
              sx={{ 
                mt: 3, 
                textAlign: 'center',
                p: 3,
                borderRadius: 2,
                backgroundColor: 'rgba(0,0,0,0.02)'
              }}
            >
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
                Image Preview
              </Typography>
              
              <Box 
                sx={{ 
                  display: 'flex',
                  justifyContent: 'center',
                  mb: 3
                }}
              >
                <Box
                  component="img"
                  src={previewUrl}
                  alt="Preview"
                  sx={{
                    maxWidth: '100%',
                    maxHeight: '300px',
                    borderRadius: 2,
                    boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                  }}
                />
              </Box>
              
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 2 }}>
                <Button 
                  variant="contained" 
                  color="primary" 
                  onClick={handleSubmit}
                  disabled={isLoading}
                  startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <AutoFixHighIcon />}
                  sx={{ 
                    px: 4,
                    py: 1,
                    borderRadius: 2,
                    textTransform: 'none',
                    fontWeight: 600,
                    boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                  }}
                >
                  {isLoading ? 'Analyzing...' : 'Analyze Image'}
                </Button>
                
                <Button 
                  variant="outlined" 
                  color="secondary" 
                  onClick={handleReset}
                  disabled={isLoading}
                  startIcon={<RestartAltIcon />}
                  sx={{ 
                    px: 3,
                    py: 1,
                    borderRadius: 2,
                    textTransform: 'none',
                    fontWeight: 500
                  }}
                >
                  Reset
                </Button>
              </Box>
            </Box>
          </Grow>
        )}

        {isLoading && (
          <Fade in={isLoading}>
            <Box 
              sx={{ 
                display: 'flex', 
                flexDirection: 'column',
                alignItems: 'center', 
                justifyContent: 'center', 
                mt: 4,
                p: 4
              }}
            >
              <CircularProgress size={60} thickness={4} />
              <Typography variant="h6" sx={{ mt: 3, fontWeight: 500 }}>
                Analyzing image...
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                Please wait while our AI detects anomalies
              </Typography>
            </Box>
          </Fade>
        )}

        {result && <ResultDisplay result={result} />}
      </Paper>
    </Container>
  );
}

export default App;