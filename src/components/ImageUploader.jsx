import React from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  useTheme, 
  Paper,
  Fade
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import ImageIcon from '@mui/icons-material/Image';

const ImageUploader = ({ onFileChange }) => {
  const theme = useTheme();
  
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onFileChange(file);
    }
  };

  return (
    <Fade in={true} timeout={800}>
      <Paper
        elevation={0}
        sx={{
          border: '2px dashed',
          borderColor: 'primary.light',
          borderRadius: 3,
          p: { xs: 3, md: 5 },
          textAlign: 'center',
          backgroundColor: 'rgba(0, 0, 0, 0.02)',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          '&:hover': {
            backgroundColor: 'rgba(25, 118, 210, 0.04)',
            borderColor: 'primary.main',
            transform: 'translateY(-4px)'
          },
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }} 
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          // Reset the value to ensure change event fires even if same file is selected
          onClick={(e) => { e.target.value = null; }}
        />
        
        <Box 
          sx={{ 
            mb: 3,
            p: 2,
            borderRadius: '50%',
            backgroundColor: 'rgba(25, 118, 210, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <ImageIcon 
            sx={{ 
              fontSize: 60, 
              color: theme.palette.primary.main
            }} 
          />
        </Box>
        
        <Typography 
          variant="h5" 
          gutterBottom
          sx={{ fontWeight: 600 }}
        >
          Upload an Image for Analysis
        </Typography>
        
        <Typography 
          variant="body1" 
          color="textSecondary"
          sx={{ maxWidth: '80%', mx: 'auto', mb: 3 }}
        >
          Drop your image here or click to browse. We support JPG, PNG, and BMP formats.
        </Typography>
        
        <Button 
          variant="contained" 
          component="span"
          startIcon={<UploadFileIcon />}
          sx={{ 
            mt: 1,
            px: 3,
            py: 1.2,
            borderRadius: 2,
            textTransform: 'none',
            fontWeight: 600,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
          }}
          onClick={(e) => {
            e.stopPropagation();
            document.getElementById('file-input').click();
          }}
        >
          Select Image
        </Button>
        
        <Typography 
          variant="caption" 
          color="textSecondary"
          sx={{ mt: 2 }}
        >
          Maximum file size: 10MB
        </Typography>
      </Paper>
    </Fade>
  );
};

export default ImageUploader;