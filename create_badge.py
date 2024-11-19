import anybadge
import re
import os

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

def ensure_svg_dir():
    """Create status-svg directory if it doesn't exist"""
    svg_dir = 'status-svg'
    if not os.path.exists(svg_dir):
        os.makedirs(svg_dir)
    return svg_dir

def remove_if_exists(file_path):
    """Remove file if it exists"""
    if os.path.exists(file_path):
        os.remove(file_path)

def create_test_badge():
    status = parse_pytest_output()
    svg_dir = ensure_svg_dir()
    
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
    
    # Set file path
    badge_path = os.path.join(svg_dir, 'test_status.svg')
    
    # Remove existing badge if it exists
    remove_if_exists(badge_path)
    
    # Save the badge
    badge.write_badge(badge_path)

def create_build_badge():
    svg_dir = ensure_svg_dir()
    try:
        # Create build badge
        badge = anybadge.Badge(
            label='build',
            value='passing',
            default_color='green'
        )
        
        # Set file path
        badge_path = os.path.join(svg_dir, 'build_status.svg')
        
        # Remove existing badge if it exists
        remove_if_exists(badge_path)
        
        # Save the badge
        badge.write_badge(badge_path)
    except Exception as e:
        # If there's an error, create a failed badge
        badge = anybadge.Badge(
            label='build',
            value='failed',
            default_color='red'
        )
        badge_path = os.path.join(svg_dir, 'build_status.svg')
        remove_if_exists(badge_path)
        badge.write_badge(badge_path)

if __name__ == '__main__':
    create_test_badge()
    create_build_badge() 