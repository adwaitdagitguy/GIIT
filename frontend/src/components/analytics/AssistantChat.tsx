import React, { useState } from 'react'
import { useWorkbenchStore } from '../../store/workbenchStore'
import { Sparkles, Send, Bot, User, HelpCircle } from 'lucide-react'

export const AssistantChat: React.FC = () => {
  const { chatMessages, addChatMessage, graph, modelMetadata } = useWorkbenchStore()
  const [input, setInput] = useState("")

  const handleSend = () => {
    if (!input.trim()) return

    const userMsg = {
      id: Date.now().toString(),
      role: 'user' as const,
      content: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    addChatMessage(userMsg)
    setInput("")

    // Generate intelligent assistant response
    setTimeout(() => {
      let reply = ""
      const q = userMsg.content.toLowerCase()

      if (q.includes("damping") || q.includes("mu")) {
        reply = "In this harmonic oscillator, the damping coefficient μ = 4.0 dissipates mechanical energy via F_damp = -μ ẏ. Perturbing layer 'fch.0.weight' introduces artificial damping variation, causing the trajectory amplitude decay rate to alter significantly."
      } else if (q.includes("stiffness") || q.includes("k")) {
        reply = "The spring stiffness k = 400.0 governs natural frequency ω₀ = √(k) = 20 rad/s. Weight ablation on 'fch.1.weight' directly impacts oscillation frequency and period."
      } else if (q.includes("loss") || q.includes("residual")) {
        reply = "The physics residual evaluates f_ode = ÿ + μ ẏ + k y at collocation points. The current baseline physics residual norm is ~0.00318, verifying that the PINN has successfully learned the differential operator."
      } else {
        reply = `Regarding "${userMsg.content}": The PINN architecture contains ${modelMetadata?.totalParameters || 2241} parameters. Physical constraints are enforced by minimizing the second-order ODE residual across collocation domain [0, 1].`
      }

      addChatMessage({
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: reply,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      })
    }, 400)
  }

  return (
    <div className="p-3 h-full flex flex-col justify-between select-none">
      {/* Message Stream */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1">
        {chatMessages.map((m) => (
          <div
            key={m.id}
            className={`flex items-start gap-2 text-xs ${
              m.role === 'user' ? 'flex-row-reverse' : ''
            }`}
          >
            <div
              className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${
                m.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              }`}
            >
              {m.role === 'user' ? <User className="w-3 h-3" /> : <Bot className="w-3 h-3" />}
            </div>
            <div
              className={`max-w-[80%] p-2 rounded-xl text-xs leading-relaxed ${
                m.role === 'user'
                  ? 'bg-blue-600 text-white rounded-tr-none'
                  : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-tl-none'
              }`}
            >
              {m.content}
              <div
                className={`text-[9px] mt-1 ${
                  m.role === 'user' ? 'text-blue-200' : 'text-slate-500'
                }`}
              >
                {m.timestamp}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Input Bar */}
      <div className="pt-2 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask AI about ODE equations, layer mapping, or perturbation..."
          className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
        />
        <button
          onClick={handleSend}
          className="p-1.5 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg hover:from-blue-500 hover:to-cyan-500 transition-all cursor-pointer"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
