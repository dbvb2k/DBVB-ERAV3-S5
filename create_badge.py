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

def create_test_badge():
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

def create_build_badge():
    try:
        # Create build badge
        badge = anybadge.Badge(
            label='build',
            value='passing',
            default_color='green'
        )
        
        # Save the badge
        badge.write_badge('build_status.svg')
    except Exception as e:
        # If there's an error, create a failed badge
        badge = anybadge.Badge(
            label='build',
            value='failed',
            default_color='red'
        )
        badge.write_badge('build_status.svg')

if __name__ == '__main__':
    create_test_badge()
    create_build_badge() 