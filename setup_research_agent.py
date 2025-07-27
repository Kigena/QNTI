#!/usr/bin/env python3
"""
QNTI Research Agent Setup Script
Installs dependencies and configures the automated research system
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing Research Agent dependencies...")
    
    try:
        # Install main requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_research.txt"])
        print("✅ Research dependencies installed successfully!")
        
        # Install Ollama (if not already installed)
        try:
            import ollama
            print("✅ Ollama already available")
        except ImportError:
            print("📦 Installing Ollama...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
            print("✅ Ollama installed!")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    return True

def setup_directories():
    """Setup research directories"""
    print("📁 Setting up research directories...")
    
    research_dir = Path("qnti_research")
    directories = [
        research_dir,
        research_dir / "raw",
        research_dir / "processed", 
        research_dir / "summaries",
        research_dir / "rag_index"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"   📂 Created {directory}")
    
    print("✅ Research directories created!")

def create_config_file():
    """Create research agent configuration"""
    print("⚙️ Creating research configuration...")
    
    config = {
        "research_agent": {
            "enabled": True,
            "research_dir": "qnti_research",
            "update_intervals": {
                "high_priority": 30,    # minutes
                "medium_priority": 120, # minutes  
                "low_priority": 360     # minutes
            },
            "rag_settings": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "similarity_top_k": 5,
                "llm_model": "llama3",
                "embedding_model": "llama3"
            },
            "sources": {
                "federal_reserve": {
                    "enabled": True,
                    "priority": 10,
                    "keywords": ["monetary policy", "interest rates", "inflation"]
                },
                "ecb": {
                    "enabled": True, 
                    "priority": 9,
                    "keywords": ["euro", "european central bank", "banking"]
                },
                "bank_of_england": {
                    "enabled": True,
                    "priority": 8, 
                    "keywords": ["gbp", "uk economy", "financial stability"]
                },
                "imf": {
                    "enabled": True,
                    "priority": 7,
                    "keywords": ["global economy", "crisis", "growth"]
                }
            },
            "filtering": {
                "min_relevance_score": 2.0,
                "max_documents_per_source": 10,
                "content_min_length": 100
            }
        }
    }
    
    config_path = Path("qnti_research_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_path}")

def test_research_agent():
    """Test the research agent setup"""
    print("🧪 Testing research agent...")
    
    try:
        from qnti_research_agent import QNTIResearchAgent
        
        # Create test instance
        agent = QNTIResearchAgent()
        print("✅ Research agent created successfully!")
        
        # Test RAG system
        if agent.rag_index is not None:
            print("✅ RAG system initialized!")
        else:
            print("⚠️ RAG system not available (may need Ollama running)")
        
        # Test database
        if agent.db_path.exists():
            print("✅ Research database ready!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing research agent: {e}")
        return False

def main():
    """Main setup process"""
    print("=" * 60)
    print("🚀 QNTI Research Agent Setup")
    print("=" * 60)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return
    
    # Step 2: Setup directories
    setup_directories()
    
    # Step 3: Create configuration
    create_config_file()
    
    # Step 4: Test setup
    if test_research_agent():
        print("\n" + "=" * 60)
        print("🎉 Research Agent Setup Complete!")
        print("=" * 60)
        print("\n📋 Next Steps:")
        print("1. Start Ollama server: ollama serve")
        print("2. Pull llama3 model: ollama pull llama3")
        print("3. Start QNTI system with research agent enabled")
        print("\n🔍 The research agent will automatically:")
        print("• Download research from Fed, ECB, BoE, IMF")
        print("• Generate AI summaries of findings")
        print("• Index content for RAG queries")
        print("• Provide research insights in market intelligence")
        print("\n💡 Research will be stored in: qnti_research/")
    else:
        print("❌ Setup completed with errors. Check logs above.")

if __name__ == "__main__":
    main() 