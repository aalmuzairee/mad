// This script sets only the initial-scale based on screen height
// Place it as early as possible in your HTML
(function() {
    // Get screen height (using various methods for cross-browser support)
    var screenHeight = window.innerHeight || document.documentElement.clientHeight || screen.height;
    
    // Default scale
    var initialScale = 1.0;
    
    // Adjust scale based on height
    if (screenHeight < 600) {
      // Very small height screens (phones in landscape, etc)
      initialScale = 1.0;
      console.log('Very small height detected: ' + screenHeight + 'px, setting scale to 1.0');
    } 
    else if (screenHeight < 800) {
      // Small laptop screens or small tablets
      initialScale = 0.9;
      console.log('Small height detected: ' + screenHeight + 'px, setting scale to 0.9');
    }
    else if (screenHeight < 900) {
      // Medium laptop screens
      initialScale = 0.85;
      console.log('Medium height detected: ' + screenHeight + 'px, setting scale to 0.85');
    }
    else if (screenHeight < 1200) {
      // Standard laptop screens
      initialScale = 0.8;
      console.log('Standard laptop height detected: ' + screenHeight + 'px, setting scale to 0.8');
    }
    else {
      // Large displays
      initialScale = 1.0;
      console.log('Large height detected: ' + screenHeight + 'px, setting scale to 1.0');
    }
    
    // Apply the viewport meta tag with ONLY the initial-scale property
    var viewportMeta = document.querySelector('meta[name="viewport"]');
    
    if (viewportMeta) {
      // Update existing viewport - only set initial-scale
      viewportMeta.setAttribute('content', 'initial-scale=' + initialScale);
      console.log('Updated existing viewport meta with scale: ' + initialScale);
    } else {
      // Create new viewport meta - only set initial-scale
      viewportMeta = document.createElement('meta');
      viewportMeta.setAttribute('name', 'viewport');
      viewportMeta.setAttribute('content', 'initial-scale=' + initialScale);
      
      // Add to head
      var head = document.head || document.getElementsByTagName('head')[0];
      head.appendChild(viewportMeta);
      console.log('Created new viewport meta with scale: ' + initialScale);
    }
  })();