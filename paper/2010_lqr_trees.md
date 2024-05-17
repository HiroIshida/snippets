### how to certify the region of attraction of a LQR controller (time invariant case)
- Use the local LQR's cost-to-go function as the Lyapunov function candidate
- Define B(\rho) as the lower-level set of the Lyapunov function
- find the largest \rho such that V is a Lyapunov function in B(\rho) 
- One can certify that B(\rho) is Lyapunov stable by Sum-of-Squares (SOS) programming (14)
- Considering that J could be approximated by taylor explansion, j(x) + h(x) * (\rho - J) + \eps x^2 <= 0 is the SOS condition with another h(x) >= 0 SOS condition
- So we could plug them into the solver to find h(x)
- To find the maximum \rho, we do just a line search on \rho and check if the SOS condition is satisfied for each \rho
