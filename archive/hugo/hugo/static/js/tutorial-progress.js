document.addEventListener('DOMContentLoaded', () => {
    const checkboxes = document.querySelectorAll('.step-complete');
    const storageKey = 'sounio-tutorial-progress';

    // Load saved progress
    const savedProgress = JSON.parse(localStorage.getItem(storageKey) || '{}');

    checkboxes.forEach((checkbox, index) => {
        const stepId = `step-${index}`;

        // Restore state
        if (savedProgress[stepId]) {
            checkbox.checked = true;
        }

        // Save on change
        checkbox.addEventListener('change', () => {
            savedProgress[stepId] = checkbox.checked;
            localStorage.setItem(storageKey, JSON.stringify(savedProgress));

            // Optional: Calculate and display progress percentage
            const totalSteps = checkboxes.length;
            const completedSteps = Object.values(savedProgress).filter(Boolean).length;
            const percentComplete = Math.round((completedSteps / totalSteps) * 100);
            console.log(`Tutorial progress: ${percentComplete}% (${completedSteps}/${totalSteps} steps completed)`);
        });
    });

    // Optional: Show initial progress
    if (checkboxes.length > 0) {
        const totalSteps = checkboxes.length;
        const completedSteps = Object.values(savedProgress).filter(Boolean).length;
        console.log(`Tutorial progress loaded: ${completedSteps}/${totalSteps} steps completed`);
    }
});
