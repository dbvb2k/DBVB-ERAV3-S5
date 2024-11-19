import anybadge
import re

def parse_pytest_output():
    try:
        with open('pytest_output.txt', 'r') as f:
            content = f.read()
            
        # Check if there were any failures
        if 'failed' in content.lower():
            return 'failed'
        elif 'passed' in content.lower():
            return 'passed'
        else:
            return 'unknown'
    except:
        return 'unknown'

def create_badge():
    status = parse_pytest_output()
    
    # Define colors for different statuses
    colors = {
        'passed': 'green',
        'failed': 'red',
        'unknown': 'gray'
    }
    
    # Create the badge
    badge = anybadge.Badge(
        label='tests',
        value=status,
        default_color=colors[status]
    )
    
    # Save the badge
    badge.write_badge('test_status.svg')

if __name__ == '__main__':
    create_badge() 