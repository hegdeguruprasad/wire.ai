import React, { useRef } from 'react';
import { Brain, BrainCircuit as Circuit, Workflow, Github, Twitter, Linkedin, Search, BarChart as FlowChart, CircuitBoard, Network as Network2, FileCheck, Upload, Settings, ChevronLeft, ChevronRight } from 'lucide-react';
import { ParticleBackground } from './components/ParticleBackground';
import { Logo } from './components/Logo';




function App() {

  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: 'left' | 'right') => {
    if (scrollContainerRef.current) {
      const scrollAmount = 400;
      scrollContainerRef.current.scrollBy({
        left: direction === 'left' ? -scrollAmount : scrollAmount,
        behavior: 'smooth'
      });
    }
  };

  const roadmapPhases = [
    {
      phase: "01",
      title: "Datasheet Q&A",
      icon: <Search className="w-8 h-8" />,
      description: "Create an application to search components and answer questions based on datasheets using RAG."
    },
    {
      phase: "02",
      title: "LLM Fine-Tuning",
      icon: <Brain className="w-8 h-8" />,
      description: "Fine-tune a large language model with global datasheets for general electronics design Q&A."
    },
    {
      phase: "03",
      title: "Flow Diagram Creation",
      icon: <FlowChart className="w-8 h-8" />,
      description: "Generate flow diagrams based on user requirements and conversations."
    },
    {
      phase: "04",
      title: "Schematic Design",
      icon: <CircuitBoard className="w-8 h-8" />,
      description: "Generate schematics exportable to ECAD tools like KiCAD or Altium."
    },
    {
      phase: "05",
      title: "PCB Layout Generation",
      icon: <Network2 className="w-8 h-8" />,
      description: "Create PCB layouts with auto-routed traces and fully functional boards."
    },
    {
      phase: "06",
      title: "Verification",
      icon: <FileCheck className="w-8 h-8" />,
      description: "Verify schematics and layouts with AI agents for accuracy."
    },
    {
      phase: "07",
      title: "Legacy Design Improvement",
      icon: <Upload className="w-8 h-8" />,
      description: "Allow users to upload existing designs and build upon them with AI."
    },
    {
      phase: "08",
      title: "Optimization",
      icon: <Settings className="w-8 h-8" />,
      description: "Improve the process with enhanced efficiency and accuracy."
    }
  ];


  return (
    <div className="min-h-screen bg-gradient-to-b from-[#00ff7f]/10 to-black text-white">
      <ParticleBackground />


      {/* Logo */}
      <div className="fixed top-0 left-0 z-50 p-6">
        <Logo />
      </div>

      {/* Hero Section */}
      <header className="min-h-screen flex flex-col items-center justify-center text-center px-4 relative">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,127,0.1)_0%,transparent_70%)]" />
        <h1 className="text-6xl md:text-8xl font-bold mb-6 animate-pulse text-[#00ff7f] [text-shadow:_0_0_30px_rgba(0,255,127,0.5)]">
          Wire.ai
        </h1>
        <p className="text-3xl md:text-4xl mb-4">The AI Revolution for Circuit Design</p>
        <p className="text-xl md:text-2xl mb-8 text-gray-300">
          From ideas to schematics in minutes – no expertise needed.
        </p>
        <button className="px-8 py-4 bg-[#00ff7f]/20 hover:bg-[#00ff7f]/30 border-2 border-[#00ff7f] rounded-full text-[#00ff7f] text-xl transition-all hover:scale-105 hover:shadow-[0_0_30px_rgba(0,255,127,0.3)] animate-pulse">
          Get Started with Wire.ai
        </button>
      </header>

      {/* About Section */}
      <section className="py-20 px-4 md:px-8 max-w-6xl mx-auto">
        <h2 className="text-4xl font-bold mb-8 text-center">About Wire.ai</h2>
        <p className="text-xl text-center text-gray-300 max-w-3xl mx-auto">
          Wire.ai empowers innovators by breaking down the barriers to hardware design. Whether you're a beginner or a professional, Wire.ai helps you transform your ideas into complete circuit diagrams.
        </p>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 md:px-8 bg-black/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold mb-12 text-center">Key Features</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: <Brain className="w-12 h-12 text-[#00ff7f]" />,
                title: "Conversational Circuit Design",
                description: "Chat with Wire.ai to define your requirements and generate circuit diagrams."
              },
              {
                icon: <Circuit className="w-12 h-12 text-[#00ff7f]" />,
                title: "AI-Powered Insights",
                description: "Leverage AI to choose components, calculate parameters, and optimize your design."
              },
              {
                icon: <Workflow className="w-12 h-12 text-[#00ff7f]" />,
                title: "Seamless Workflow",
                description: "Turn high-level ideas into actionable schematics, ready for prototyping."
              }
            ].map((feature, index) => (
              <div key={index} className="p-6 rounded-xl bg-gradient-to-b from-[#00ff7f]/10 to-transparent border border-[#00ff7f]/20 hover:border-[#00ff7f]/40 transition-all">
                <div className="mb-4">{feature.icon}</div>
                <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                <p className="text-gray-300">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-20 px-4 md:px-8">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold mb-12 text-center">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: "01", title: "Enter your idea", description: "Describe your requirements in plain English" },
              { step: "02", title: "AI Analysis", description: "Wire.ai suggests components and generates a flow diagram" },
              { step: "03", title: "Final Schematic", description: "Get a complete circuit schematic ready for manufacturing" }
            ].map((step, index) => (
              <div key={index} className="relative">
                <div className="text-6xl font-bold text-[#00ff7f]/10 absolute -top-8 left-0">{step.step}</div>
                <h3 className="text-xl font-bold mb-2 relative">{step.title}</h3>
                <p className="text-gray-300">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section className="py-20 px-4 md:px-8 bg-black/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold mb-12 text-center">Use Cases</h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { title: "Prototyping for Startups", description: "Accelerate product development timelines." },
              { title: "Education", description: "Empower students to learn by designing their own circuits." },
              { title: "Professional Engineers", description: "Streamline repetitive design tasks and focus on innovation." }
            ].map((useCase, index) => (
              <div key={index} className="p-6 rounded-xl bg-[#00ff7f]/5 hover:bg-[#00ff7f]/10 transition-all">
                <h3 className="text-xl font-bold mb-2">{useCase.title}</h3>
                <p className="text-gray-300">{useCase.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section
      <section className="py-20 px-4 md:px-8 text-center">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-4xl font-bold mb-8">Revolutionize Your Electronics Design Process Today</h2>
          <div className="flex flex-col md:flex-row gap-4 justify-center">
            <button className="px-8 py-4 bg-[#00ff7f]/20 hover:bg-[#00ff7f]/30 border-2 border-[#00ff7f] rounded-full text-[#00ff7f] transition-all hover:scale-105">
              Start Your Free Trial
            </button>
            <button className="px-8 py-4 bg-transparent hover:bg-white/5 border-2 border-white rounded-full transition-all hover:scale-105">
              Learn More
            </button>
          </div>
        </div>
      </section> */}

      {/* Roadmap Section */}
      <section className="py-20 px-4 md:px-8 bg-gradient-to-r from-[#00ff7f]/5 via-black to-[#00ff7f]/5 relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,255,127,0.05)_0%,transparent_70%)]" />
        <div className="max-w-6xl mx-auto mb-12">
          <h2 className="text-4xl font-bold text-center mb-4">Development Roadmap</h2>
          <p className="text-xl text-center text-gray-300">Our journey to revolutionize circuit design</p>
        </div>
        
        <div className="relative max-w-6xl mx-auto">
          {/* Navigation Buttons */}
          <button 
            onClick={() => scroll('left')} 
            className="absolute left-0 top-1/2 -translate-y-1/2 z-10 p-2 bg-black/50 rounded-full border border-[#00ff7f]/30 text-[#00ff7f] hover:bg-[#00ff7f]/20 transition-all"
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
          <button 
            onClick={() => scroll('right')} 
            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 p-2 bg-black/50 rounded-full border border-[#00ff7f]/30 text-[#00ff7f] hover:bg-[#00ff7f]/20 transition-all"
          >
            <ChevronRight className="w-6 h-6" />
          </button>

          {/* Scrollable Container */}
          <div 
            ref={scrollContainerRef}
            className="overflow-x-auto hide-scrollbar relative flex gap-6 px-12 pb-4 snap-x snap-mandatory"
          >
            {/* Timeline Line */}
            <div className="absolute top-[45%] left-14 right-14 h-0.5 bg-gradient-to-r from-transparent via-[#00ff7f]/30 to-transparent" />
            
            {roadmapPhases.map((phase, index) => (
              <div 
                key={index}
                className="snap-center flex-none w-[300px] h-[220px] group"
              >
                <div className="relative h-full bg-black/40 p-6 rounded-xl border border-[#00ff7f]/20 backdrop-blur-sm transition-all duration-300 hover:border-[#00ff7f]/40 hover:shadow-[0_0_30px_rgba(0,255,127,0.1)] hover:scale-105 flex flex-col">
                  {/* Phase Number */}
                  <div className="absolute -top-3 -left-3 w-10 h-10 rounded-full bg-[#00ff7f]/10 border border-[#00ff7f]/30 flex items-center justify-center text-[#00ff7f] font-bold">
                    {phase.phase}
                  </div>
                  
                  {/* Icon */}
                  <div className="mb-4 text-[#00ff7f]">
                    {phase.icon}
                  </div>
                  
                  {/* Content */}
                  <div className="flex-1 flex flex-col">
                    <h3 className="text-xl font-bold mb-2 text-[#00ff7f]">
                      {phase.title}
                    </h3>
                    <p className="text-gray-300 text-sm transition-all duration-300 group-hover:text-white flex-1">
                      {phase.description}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 md:px-8 border-t border-white/10">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <h3 className="text-2xl font-bold text-[#00ff7f] mb-4">Wire.ai</h3>
              <p className="text-gray-300">The future of circuit design</p>
            </div>
            <div>
              <h4 className="font-bold mb-4">Links</h4>
              <ul className="space-y-2 text-gray-300">
                <li><a href="#" className="hover:text-[#00ff7f]">About Wire.ai</a></li>
                <li><a href="#" className="hover:text-[#00ff7f]">FAQs</a></li>
                <li><a href="#" className="hover:text-[#00ff7f]">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-[#00ff7f]">Contact Us</a></li>
              </ul>
            </div>
            <div className="md:col-span-2">
              <h4 className="font-bold mb-4">Stay Updated</h4>
              <div className="flex gap-2">
                <input
                  type="email"
                  placeholder="Enter your email to stay updated"
                  className="flex-1 px-4 py-2 bg-white/5 rounded-lg border border-white/20 focus:border-[#00ff7f] focus:outline-none"
                />
                <button className="px-6 py-2 bg-[#00ff7f]/20 hover:bg-[#00ff7f]/30 border border-[#00ff7f] rounded-lg text-[#00ff7f]">
                  Subscribe
                </button>
              </div>
              {/* <div className="flex gap-4 mt-6">
                <a href="#" className="text-gray-300 hover:text-[#00ff7f]">
                  <Twitter className="w-6 h-6" />
                </a>
                <a href="#" className="text-gray-300 hover:text-[#00ff7f]">
                  <Github className="w-6 h-6" />
                </a>
                <a href="#" className="text-gray-300 hover:text-[#00ff7f]">
                  <Linkedin className="w-6 h-6" />
                </a>
              </div> */}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;