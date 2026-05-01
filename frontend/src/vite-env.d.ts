declare module '*.css' {
  const content: { [className: string]: string }
  export default content
}

declare module 'react-syntax-highlighter/dist/esm/styles/prism' {
  const content: any
  export default content
}
