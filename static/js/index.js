window.HELP_IMPROVE_VIDEOJS = false;

// Define task lists for each environment
const taskData = {
  'MetaWorld': ['Disassemble', 'Hammer', 'Shelf Place', 'Stick Push', 'Hand Insert'],
  'ManiSkill3': ['PokeCube', 'PlaceSphere', 'PickCube', 'PushCube', 'PullCube']
};

// Keep track of the current environment and task
let currentEnv = 'MetaWorld';  // Changed from 'ManiSkill3' to 'MetaWorld'
let currentTaskIndex = 0;

// Function to switch the environment (top-level tab)
function switchEnv(envName) {
  // Don't do anything if we're already on this environment
  if (currentEnv === envName) return;
  
  // Get the current task before switching
  const oldEnv = currentEnv;
  const oldTask = taskData[oldEnv][currentTaskIndex];
  
  // Update active tab styling for environment tabs
  const envTabs = document.getElementById('env-tabs').querySelectorAll('.tab');
  envTabs.forEach(tab => {
    if (tab.textContent === envName) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });
  
  // Update the task tabs for the new environment
  const taskTabs = document.getElementById('task-tabs');
  taskTabs.innerHTML = ''; // Clear existing tabs
  
  // Create new task tabs
  taskData[envName].forEach((task, index) => {
    const button = document.createElement('button');
    button.className = index === 0 ? 'tab active' : 'tab';
    button.textContent = task;
    button.onclick = () => switchTask(index);
    taskTabs.appendChild(button);
  });
  
  // Update current environment and reset task index to 0
  currentEnv = envName;
  currentTaskIndex = 0;
  const newTask = taskData[currentEnv][0];
  
  // Update video sources for the new environment and task
  updateVideoSources(oldEnv, oldTask, currentEnv, newTask);
}

// Function to switch the task (bottom-level tab)
function switchTask(taskIndex) {
  // Don't do anything if we're already on this task
  if (currentTaskIndex === taskIndex) return;
  
  // Update active tab styling for task tabs
  const taskTabs = document.getElementById('task-tabs').querySelectorAll('.tab');
  taskTabs.forEach((tab, index) => {
    if (index === taskIndex) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });
  
  // Store the old task index and name before updating
  const oldTaskIndex = currentTaskIndex;
  const oldTask = taskData[currentEnv][oldTaskIndex];
  
  // Update current task index and get new task name
  currentTaskIndex = taskIndex;
  const newTask = taskData[currentEnv][taskIndex];
  
  // Update video sources for the new task (same environment)
  updateVideoSources(currentEnv, oldTask, currentEnv, newTask);
}

// Helper function to convert task name to filename format
function taskNameToFilename(envName, taskName) {
  if (envName === 'ManiSkill3') {
    // For ManiSkill3, use the exact same name
    return taskName;
  } else if (envName === 'MetaWorld') {
    // For MetaWorld, convert to lowercase and replace spaces with hyphens
    return taskName.toLowerCase().replace(/ /g, '-');
  }
  return taskName; // Default fallback
}

// Function to update all video sources when environment or task changes
function updateVideoSources(oldEnv, oldTask, newEnv, newTask) {
  const videoElements = document.querySelectorAll('.video-container video');
  
  // Convert task names to their corresponding filename formats
  const oldTaskFilename = taskNameToFilename(oldEnv, oldTask);
  const newTaskFilename = taskNameToFilename(newEnv, newTask);
  
  videoElements.forEach((video) => {
    const currentSrc = video.src;
    let newSrc = currentSrc;
    
    // If we're changing environments, we need to update both the task name and potentially the folder structure
    if (oldEnv !== newEnv) {
      // Change env directory
      if (newEnv === 'ManiSkill3'){
        newSrc = newSrc.replace('/mw/','/ms/')
      }
      else {
        newSrc = newSrc.replace('/ms/','/mw/')
      }

      // Replace the old task with the new task in filename format
      newSrc = newSrc.replace(oldTaskFilename, newTaskFilename);

    } else {
      // Just replace the task name (we're in the same environment)
      newSrc = newSrc.replace(oldTaskFilename, newTaskFilename);
    }
    
    // Update the video source
    video.src = newSrc;
    
    // Force video reload
    video.load();
    video.play().catch(e => {
      console.warn('Auto-play prevented:', e);
    });
  });
}

// Initialize the first environment and task when the page loads
document.addEventListener('DOMContentLoaded', function() {
  // Set initial environment and task
  currentEnv = 'MetaWorld';  // Changed from 'ManiSkill3' to 'MetaWorld'
  currentTaskIndex = 0;
  
  // Make sure the correct environment tab is active
  const envTabs = document.getElementById('env-tabs').querySelectorAll('.tab');
  envTabs.forEach(tab => {
    if (tab.textContent === currentEnv) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });
  
  // Set up initial task tabs
  const taskTabs = document.getElementById('task-tabs');
  taskTabs.innerHTML = ''; // Clear any existing tabs
  
  // Create task tabs for the initial environment
  taskData[currentEnv].forEach((task, index) => {
    const button = document.createElement('button');
    button.className = index === 0 ? 'tab active' : 'tab';
    button.textContent = task;
    button.onclick = () => switchTask(index);
    taskTabs.appendChild(button);
  });

  // Back to Top Button
  const backToTopButton = document.getElementById('back-to-top');
  if (backToTopButton) {
    backToTopButton.addEventListener('click', function() {
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    });
  } else {
    console.error("Back to top button not found in the DOM");
  }
});

