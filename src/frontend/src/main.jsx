import React from 'react';
import ReactDOM from 'react-dom/client';
import Chat from './Chat';
import './index.css'; // Ensure this line is included

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Chat />
  </React.StrictMode>
);
