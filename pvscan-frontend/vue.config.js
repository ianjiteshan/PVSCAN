const path = require('path');

module.exports = {
  outputDir: path.resolve(__dirname, '../pvscan_backend/src/static'),
  indexPath: path.resolve(__dirname, '../pvscan_backend/src/templates/index.html'),
  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000', // Flask backend URL during local dev
        changeOrigin: true
      }
    }
  }
};
