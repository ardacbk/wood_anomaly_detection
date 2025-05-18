import React from 'react';
import { Box, Button, Typography } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const ImageUploader = ({ onFileChange }) => {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onFileChange(file);
    }
  };

  return (
    <Box 
      sx={{
        border: '2px dashed #ccc',
        borderRadius: 2,
        p: 4,
        textAlign: 'center',
        backgroundColor: '#f8f9fa',
        cursor: 'pointer',
        '&:hover': {
          backgroundColor: '#f0f0f0',
          borderColor: '#aaa',
        }
      }} 
      onClick={() => document.getElementById('file-input').click()}
    >
      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
      
      <CloudUploadIcon sx={{ fontSize: 60, color: '#3f51b5', mb: 2 }} />
      
      <Typography variant="h6" gutterBottom>
        Click or Drag to Upload Image
      </Typography>
      
      <Typography variant="body2" color="textSecondary">
        Supported formats: JPG, PNG, BMP
      </Typography>
      
      <Button 
        variant="contained" 
        component="span"
        sx={{ mt: 2 }}
        onClick={(e) => {
          e.stopPropagation();
          document.getElementById('file-input').click();
        }}
      >
        Select File
      </Button>
    </Box>
  );
};

export default ImageUploader;